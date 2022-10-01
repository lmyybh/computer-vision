import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import time
import logging

import toml
import torch

from yolo.data import build_dataloader
from yolo.v3 import YoloV3


class Builder:
    def __init__(self, config_file):
        self.cfg = toml.load(config_file)
        self.device = torch.device(self.cfg["TRAIN"]["DEVICE"])
        self.start_iter = self.cfg["TRAIN"]["START_ITER"]
        self.num_iters = self.cfg["TRAIN"]["NUM_ITERS"]
        self.log_per_iters = self.cfg["TRAIN"]["LOG_PER_ITERS"]
        self.checkpoints_per_iters = self.cfg["TRAIN"]["CHECKPOINT_PER_ITERS"]

    def build_model(self):
        model = YoloV3(self.cfg)
        path = self.cfg["TRAIN"]["PRETRAINED_WEIGHTS_PATH"]
        if os.path.exists(path):
            model.load_state_dict(torch.load(path))

        return model.to(self.device)

    def make_optimizer(self, model):
        return torch.optim.SGD(
            model.parameters(),
            lr=self.cfg["TRAIN"]["OPTIMIZER"]["BASE_LR"],
            momentum=self.cfg["TRAIN"]["OPTIMIZER"]["MOMENTUM"],
            weight_decay=self.cfg["TRAIN"]["OPTIMIZER"]["WEIGHT_DECAY"],
        )

    def make_lr_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=self.cfg["TRAIN"]["SCHEDULER"]["STEP_SIZE"], 
            gamma=self.cfg["TRAIN"]["SCHEDULER"]["GAMMA"], 
            last_epoch=self.start_iter - 1
        )

    def make_output_dir(self, subdirs=["checkpoints"]):
        output_dir = os.path.join(
            self.cfg["OUTPUT"],
            "test",  # time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()),
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for subdir in subdirs:
            dirname = os.path.join(output_dir, subdir)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
        return output_dir

    def make_logger(self, output_dir):
        log_file = os.path.join(output_dir, "log.txt")
        logger = logging.getLogger("yolov3_logger")
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s- %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger


class Trainer:
    def __init__(
        self, model, optimizer, lr_scheduler, logger, device, output_dir
    ):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.logger = logger
        self.device = device
        self.output_dir = output_dir

    def save_checkpoint(self, name):
        filename = os.path.join(self.output_dir, "checkpoints", name)
        torch.save(self.model.state_dict(), filename)
        self.logger.info(f"Save checkpoint: {filename}")

    def test(self, dataloader):
        self.model.eval()
        mean_loss_dict = {}
        mean_losses = 0

        for i, (images, targets) in enumerate(dataloader):
            images = images.to(self.device)
            targets = [target.to(self.device) for target in targets]
            with torch.no_grad(): # 必须有，负责会梯度累计，超显存
                loss_dict = self.model(images, targets)
            for k, v in loss_dict.items():
                if k not in mean_loss_dict:
                    mean_loss_dict[k] = v.item()
                else:
                    mean_loss_dict[k] += v.item()
            mean_losses += sum(loss.item() for loss in loss_dict.values())
        
        n = len(test_dataloader)
        mean_loss_dict = {k: v / n for k, v in mean_loss_dict.items()}
        mean_losses /= n

        return mean_losses, mean_loss_dict

    def iters_train_and_test(self, train_dataloader, test_dataloader, start_iter, num_iters, log_per_iters, checkpoints_per_iters):
        self.logger.info("Start training")

        iter = start_iter
        while iter < num_iters:
            for i, (images, targets) in enumerate(train_dataloader):
                # train
                self.model.train()
                images = images.to(self.device)
                targets = [target.to(self.device) for target in targets]
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()

                if (iter + 1) % log_per_iters == 0:
                    losses = losses.item()
                    loss_dict = {k: v.item() for k, v in loss_dict.items()}
                    self.logger.info(
                        f"train: [iters: {iter}] [lr: {self.lr_scheduler.get_last_lr()[-1]}] [loss: {losses}] [loss_dict:{loss_dict}]"
                    )

                if (iter + 1) % checkpoints_per_iters == 0:
                    self.save_checkpoint(f"iters_{iter}.pt")

                    # test
                    mean_losses, mean_loss_dict = self.test(test_dataloader)

                    self.logger.info(
                        f"test: [iters: {iter}] [mean_loss: {mean_losses}] [mean_loss_dict:{mean_loss_dict}]"
                    )

                self.lr_scheduler.step()
                iter += 1
        
        self.logger.info("Finish training")


if __name__ == "__main__":
    # builder
    config_file = "/home/cgl/projects/computer-vision/yolo/yolo/configs/yolov3.toml"
    builder = Builder(config_file)
    model = builder.build_model()
    optimizer = builder.make_optimizer(model)
    lr_scheduler = builder.make_lr_scheduler(optimizer)
    output_dir = builder.make_output_dir()
    logger = builder.make_logger(output_dir)
    device = builder.device

    trainer = Trainer(
        model, optimizer, lr_scheduler, logger, builder.device, output_dir
    )

    logger.info(f"Output directory: {output_dir}")

    # modules
    train_dataloader = build_dataloader(builder.cfg, is_train=True)
    test_dataloader = build_dataloader(builder.cfg, is_train=False)

    # train
    trainer.iters_train_and_test(
        train_dataloader,
        test_dataloader,
        start_iter=builder.start_iter,
        num_iters=builder.num_iters,
        log_per_iters=builder.log_per_iters,
        checkpoints_per_iters=builder.checkpoints_per_iters,
    )
