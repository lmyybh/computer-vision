import torch
import torch.nn as nn

from .backbone import build_backbone
from .rpn import build_rpn
from .roi_heads import build_roi_heads


class MaskRCNN(nn.Module):
    def __init__(self, cfg):
        super(MaskRCNN, self).__init__()
        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

    def forward(self, images, targets=None):
        features = self.backbone(images.tensors)
        proposals, proposals_losses = self.rpn(images, features, targets)
        x, detections, detector_losses = self.roi_heads(features, proposals, targets)
        if self.training:
            losses = {}
            losses.update(proposals_losses)
            losses.update(detector_losses)
            return losses
        else:
            return detections
