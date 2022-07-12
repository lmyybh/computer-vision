from collections import OrderedDict
import torch.nn as nn

from .resnet import ResNet101
from .fpn import FPN


def build_backbone(cfg):
    body = ResNet101()
    fpn = FPN(
        in_channels_list=cfg["MODEL"]["FPN"]["IN_CHANNELS_LIST"],
        out_channels=cfg["MODEL"]["FPN"]["OUT_CHANNELS"],
    )

    return nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
