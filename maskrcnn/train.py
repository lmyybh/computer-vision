import os
import time
import logging

import toml
import torch

from maskrcnn.data import build_dataloader
from maskrcnn.models.model import MaskRCNN


class Builder:
    def __init__(self, config_file):
        self.cfg = toml.load(config_file)
        self.device = torch.device(self.cfg["TRAIN"]["DEVICE"])
        self.start_epoch = self.cfg["TRAIN"]["START_EPOCH"]
        self.num_epoches = self.cfg["TRAIN"]["NUM_EPOCHES"]

    def build_model(self):
        model = MaskRCNN(self.cfg)
        path = self.cfg["TRAIN"]["PRETRAINED_WEIGHTS_PATH"]
        if os.path.exists(path):
            model.load_state_dict(torch.load(path))

        return model.to(self.device)

    def make_optimizer(self, model):
        # return torch.optim.SGD(
        #     [{"params": model.parameters(), "initial_lr": 0.001}],
        #     lr=self.cfg["SOLVER"]["INITIAL_LR"],
        #     momentum=0.9,
        #     weight_decay=0.0005,
        # )
        return torch.optim.Adam(
            model.parameters(), amsgrad=True
        )

    # def make_lr_scheduler(self, optimizer):
    #     return torch.optim.lr_scheduler.StepLR(
    #         optimizer,
    #         step_size=10,
    #         gamma=0.5,
    #         last_epoch=self.start_epoch - 1,
    #         verbose=False,
    #     )

    def make_output_dir(self, subdirs=["checkpoints"]):
        output_dir = os.path.join(
            self.cfg["OUTPUT"],
            time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()),
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
        logger = logging.getLogger("maskrcnn_logger")
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s- %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger


class Trainer:
    def __init__(
        self, model, optimizer, logger, device, output_dir, num_logs
        # self, model, optimizer, lr_scheduler, logger, device, output_dir, num_logs
    ):
        self.model = model
        self.optimizer = optimizer
        # self.lr_scheduler = lr_scheduler
        self.logger = logger
        self.device = device
        self.output_dir = output_dir
        self.num_logs = num_logs

    def save_checkpoint(self, epoch):
        filename = os.path.join(self.output_dir, "checkpoints", f"epoch_{epoch}.pt")
        torch.save(self.model.state_dict(), filename)

    def train(self, dataloader, train_mode=0):
        self.model.train()

        num_iters = len(dataloader)
        a, b = num_iters // self.num_logs, num_iters % self.num_logs
        step = a - 1 + (b > 0)
        b -= 1
        for i, (images, targets) in enumerate(dataloader):
            images = images.to(self.device)
            targets = [target.to(self.device) for target in targets]
            result = self.model(images, targets, train_mode=train_mode)
            loss_dict = result["loss_dict"]
            losses = sum(loss for loss in loss_dict.values())

            if i == step:
                item_dict = {k: v.item() for k, v in loss_dict.items()}
                self.logger.info(
                    f"train: [iters: {i}] [train_mode: {train_mode}] [loss: {losses.item()}] [loss_dict:{item_dict}]"
                )
                step += a + (b > 0)
                b -= 1

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()
        # self.lr_scheduler.step()

        return losses.item(), {k: v.item() for k, v in loss_dict.items()}

    def test(self, dataloader):
        self.model.eval()

        mean_loss_dict = {}
        mean_losses = 0

        for i, (images, targets) in enumerate(dataloader):
            images = images.to(self.device)
            targets = [target.to(self.device) for target in targets]
            loss_dict = self.model(images, targets, train_mode=2)["loss_dict"]
            for k, v in loss_dict.items():
                if k not in mean_loss_dict:
                    mean_loss_dict[k] = v.item()
                else:
                    mean_loss_dict[k] += v.item()

            mean_losses += sum(loss.item() for loss in loss_dict.values())
        n = len(dataloader)
        mean_loss_dict = {k: v / n for k, v in mean_loss_dict.items()}
        mean_losses /= n

        return mean_losses, mean_loss_dict

    def train_and_test(
        self,
        train_dataloader,
        test_dataloader,
        num_epoches,
        start_epoch=0,
        checkpoint_per_epoches=1,
        mode_steps=[5, 10],
    ):
        self.logger.info("Start training")

        train_mode = 0

        for epoch in range(start_epoch, num_epoches):

            if epoch <= mode_steps[0]:
                train_mode = 0
            elif mode_steps[0] < epoch <= mode_steps[1]:
                train_mode = 1
            else:
                train_mode = 2

            losses, loss_dict = self.train(train_dataloader, train_mode=train_mode)
            self.logger.info(
                f"train: [epoch: {epoch+1}] [train_mode: {train_mode}] [loss: {losses}] [loss_dict:{loss_dict}]"
            )

            mean_losses, mean_loss_dict = self.test(test_dataloader)
            self.logger.info(
                f"test: [epoch:{epoch+1}] [mean_loss: {mean_losses}] [mean_loss_dict:{mean_loss_dict}]"
            )

            if (epoch + 1) % checkpoint_per_epoches == 0:
                self.save_checkpoint(epoch)

        self.logger.info("Finish training")


if __name__ == "__main__":
    # builder
    config_file = (
        "/home/cgl/projects/computer-vision/maskrcnn/maskrcnn/configs/config.toml"
    )
    builder = Builder(config_file)
    model = builder.build_model()
    optimizer = builder.make_optimizer(model)
    # lr_scheduler = builder.make_lr_scheduler(optimizer)
    output_dir = builder.make_output_dir()
    logger = builder.make_logger(output_dir)
    device = builder.device
    num_logs = builder.cfg["TRAIN"]["NUM_LOGS_PER_EPOCH"]

    trainer = Trainer(
        model, optimizer, logger, builder.device, output_dir, num_logs
        # model, optimizer, lr_scheduler, logger, builder.device, output_dir, num_logs
    )

    logger.info(f"Output directory: {output_dir}")

    # modules
    train_dataloader = build_dataloader(builder.cfg, is_train=True)
    test_dataloader = build_dataloader(builder.cfg, is_train=False)

    # train
    trainer.train_and_test(
        train_dataloader,
        test_dataloader,
        num_epoches=builder.num_epoches,
        start_epoch=builder.start_epoch,
        checkpoint_per_epoches=builder.cfg["TRAIN"]["CHECKPOINT_PER_EPOCHES"],
    )
