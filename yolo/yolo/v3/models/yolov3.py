import torch
import torch.nn as nn

from .detector import build_detector


class YoloV3(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.detector = build_detector(cfg)

    def forward(self, images, targets=None):
        features = self.detector(images)
        return features
