import torch
import torch.nn as nn
import torch.nn.functional as F

from .anchor_generator import AnchorGenerator


class RPNHead(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

        # 1*1 卷积相当于全连接
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)

        for layer in [self.conv, self.cls_logits, self.bbox_pred]:
            torch.nn.init.normal_(layer.weight, std=0.01)
            torch.nn.init.constant_(layer.bias, 0)

    def forward(self, features):
        logits = []
        bbox_reg = []
        for feature in features:
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg


class RPN(nn.Module):
    def __init__(
        self,
        in_channels,
        strides=[4, 8, 16, 32, 64],
        sizes=[32, 64, 128, 256, 512],
        ratios=[0.5, 1, 2],
        training=True
    ):
        super(RPN, self).__init__()
        self.anchor_generator = AnchorGenerator(strides, sizes, ratios)
        self.head = RPNHead(
            in_channels, self.anchor_generator.num_anchors_per_location()
        )

        self.training = training

    def forward(self, images, features, targets=None):
        logits, bbox_reg = self.head(features)
        anchors = self.anchor_generator(images, features)

        return logits, bbox_reg, anchors

        # if self.training:
        #     return self._forward_train(anchors, logits, bbox_reg, targets)
        # else:
        #     return self._forward_test(anchors, logits, bbox_reg)

    def _forward_train(self, anchors, logits, bbox_reg, targets):
        pass

    def _forward_test(self, anchors, logits, bbox_reg):
        pass


def build_rpn(cfg):
    return RPN(
        cfg["MODEL"]["FPN"]["OUT_CHANNELS"],
        strides=cfg["MODEL"]["RPN"]["ANCHORS_STRIDES"],
        sizes=cfg["MODEL"]["RPN"]["ANCHORS_SIZES"],
        ratios=cfg["MODEL"]["RPN"]["ANCHORS_RATIOS"],
    )
