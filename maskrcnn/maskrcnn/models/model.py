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

    def forward(self, images, targets=None, train_mode=2):
        assert train_mode in [0, 1, 2]

        features = self.backbone(images.tensors)

        if train_mode == 0:
            proposals, proposals_losses = self.rpn(images, features, targets)
            with torch.no_grad():
                x, detections, detector_losses = self.roi_heads(
                    features, proposals, targets
                )
        elif train_mode == 1:
            with torch.no_grad():
                proposals, proposals_losses = self.rpn(images, features, targets)

            x, detections, detector_losses = self.roi_heads(
                features, proposals, targets
            )
        else:
            proposals, proposals_losses = self.rpn(images, features, targets)
            x, detections, detector_losses = self.roi_heads(
                features, proposals, targets
            )

        loss_dict = {}
        loss_dict.update(proposals_losses)
        loss_dict.update(detector_losses)

        return {"detections": detections, "loss_dict": loss_dict}
