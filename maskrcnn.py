import toml
import torch
import numpy as np

from maskrcnn.data import build_dataset, build_dataloader
from maskrcnn.models.backbone import build_backbone
from maskrcnn.models.rpn import build_rpn

cfg = toml.load("/home/cgl/projects/computer-vision/maskrcnn/configs/config.toml")

dataset = build_dataset(cfg)
dataloader = build_dataloader(dataset, cfg, is_train=True)

device = torch.device("cuda:0")
backbone = build_backbone(cfg).to(device)
rpn = build_rpn(cfg).to(device)

for images_list, targets in dataloader:
    images_list = images_list.to(device)
    targets = [target.to(device) for target in targets]

    features = backbone(images_list.tensors)

    rpn(images_list, features, targets)

    break
