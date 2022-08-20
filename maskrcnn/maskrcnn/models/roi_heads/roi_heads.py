import torch
import torch.nn as nn

from .box_head.box_head import build_roi_box_head


class ROIHeads(nn.Module):
    def __init__(self, cfg, in_channels):
        super(ROIHeads, self).__init__()
        self.box = build_roi_box_head(cfg, in_channels)

    def forward(self, features, proposals, targets=None):
        losses = {}
        x, detections, loss_box = self.box(features, proposals, targets)
        losses.update(loss_box)
        return x, detections, losses


def build_roi_heads(cfg, in_channels):
    return ROIHeads(cfg, in_channels)
