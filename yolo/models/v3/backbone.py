import torch
import torch.nn as nn

from .darknet import darknet53


class Backbone(nn.Module):
    def __init__(self, weight_path):
        super().__init__()
        self.darknet = darknet53(1000)
        self.darknet.load_state_dict(torch.load(weight_path)["state_dict"])

    def forward(self, x):
        features = []

        out = self.darknet.conv1(x)
        out = self.darknet.conv2(out)
        out = self.darknet.residual_block1(out)
        out = self.darknet.conv3(out)
        out = self.darknet.residual_block2(out)
        out = self.darknet.conv4(out)
        out = self.darknet.residual_block3(out)
        features.append(out)  # [N, 256, 52, 52]
        out = self.darknet.conv5(out)
        out = self.darknet.residual_block4(out)
        features.append(out)  # [N, 512, 26, 26]
        out = self.darknet.conv6(out)
        out = self.darknet.residual_block5(out)
        features.append(out)  # [N, 1024, 13, 13]

        return features


def build_backbone(cfg):
    return Backbone(cfg["MODEL"]["DARKENT53_WEIGTH_PATH"])
