import torch
import torch.nn as nn

from .detector import build_detector
from ..loss import build_yolov3_loss


class YoloV3(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.detector = build_detector(cfg)
        self.loss = build_yolov3_loss(cfg)

    def forward(self, images, targets=None):
        features = self.detector(images)

        if targets is not None:
            losses_dict = self.loss(features, targets)
            return losses_dict

        return features
