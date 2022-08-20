import os
import time
import logging

import toml
import torch

from maskrcnn.data import build_dataset, build_dataloader
from maskrcnn.models.model import MaskRCNN


def build_model(cfg, device):
    return MaskRCNN(cfg).to(device)


def make_optimizer(cfg, model):
    return torch.optim.SGD(
        model.parameters(),
        lr=cfg["SOLVER"]["BASE_LR"],
        momentum=cfg["SOLVER"]["MOMENTUM"],
        weight_decay=cfg["SOLVER"]["WEIGHT_DECAY"],
    )


def make_lr_scheduler(cfg, optimizer):
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg["SOLVER"]["MILESTONES"], gamma=cfg["SOLVER"]["GAMMA"]
    )


def make_output_dir(cfg, subdirs=["checkpoints"]):
    output_dir = os.path.join(
        cfg["OUTPUT"], time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for subdir in subdirs:
        dirname = os.path.join(output_dir, subdir)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
    return output_dir


def make_logger(output_dir):
    log_file = os.path.join(output_dir, "log.txt")
    logger = logging.getLogger("maskrcnn_logger")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s- %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def save_checkpoint(model, output_dir, epoch):
    filename = os.path.join(output_dir, "checkpoints", f"epoch_{epoch}.pt")
    torch.save(model.state_dict(), filename)


# config
cfg = toml.load(
    "/home/cgl/projects/computer-vision/maskrcnn/maskrcnn/configs/config.toml"
)
output_dir = make_output_dir(cfg, subdirs=["checkpoints"])
logger = make_logger(output_dir)
logger.info(f"Output directory: {output_dir}")

device = torch.device(cfg["TRAIN"]["DEVICE"])

# modules
dataset = build_dataset(cfg, is_train=True)
dataloader = build_dataloader(dataset, cfg, is_train=True)

model = build_model(cfg, device)
optimizer = make_optimizer(cfg, model)
lr_scheduler = make_lr_scheduler(cfg, optimizer)

# train
num_epoches = cfg["TRAIN"]["EPOCHES"]
log_per_period = cfg["TRAIN"]["LOG_PER_PERIOD"]
checkpoint_per_epoches = cfg["TRAIN"]["CHECKPOINT_PER_EPOCHES"]
num_iters_per_epoches = len(dataloader)

logger.info("Start training")
for epoch in range(num_epoches):
    for i, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        targets = [target.to(device) for target in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if (epoch * num_iters_per_epoches + i + 1) % log_per_period == 0:
            cpu_loss_dict = {k: v.item() for k, v in loss_dict.items()}
            logger.info(
                f"epoch: {epoch+1}, loss: {losses.item()}, loss_dict: {cpu_loss_dict}"
            )

    lr_scheduler.step()

    if (epoch + 1) % checkpoint_per_epoches == 0:
        save_checkpoint(model, output_dir, epoch)

logger.info("Finish training")
