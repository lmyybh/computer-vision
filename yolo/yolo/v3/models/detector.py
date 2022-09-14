import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone


def DBL(in_num, out_num, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(
            in_num,
            out_num,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        ),
        nn.BatchNorm2d(out_num),
        nn.LeakyReLU(),
    )


class Upsample(nn.Module):
    def __init__(self, scale_factor, mode="nearest"):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


def ConvSet(in_num, out_num):
    return nn.Sequential(
        DBL(in_num, out_num, kernel_size=1, padding=0),
        DBL(out_num, out_num, kernel_size=3, padding=1),
        DBL(out_num, out_num, kernel_size=1, padding=0),
        DBL(out_num, out_num, kernel_size=3, padding=1),
        DBL(out_num, out_num, kernel_size=1, padding=0),
    )


def ConvUp(in_num, out_num):
    return nn.Sequential(DBL(in_num, out_num, kernel_size=1, padding=0), Upsample(2))


def OutConv(in_num, out_num):
    return nn.Sequential(
        DBL(in_num, out_num, kernel_size=3, padding=1),
        DBL(out_num, out_num, kernel_size=1, padding=0),
    )


class Detector(nn.Module):
    def __init__(self, cfg, num_boxes=3, num_classes=20):
        super().__init__()
        self.num_boxes = num_boxes

        self.backbone = build_backbone(cfg)

        out_nums = num_boxes * (5 + num_classes)

        self.convset_c2 = ConvSet(1024, 1024)
        self.outconv_c2 = OutConv(1024, out_nums)
        self.convup_c21 = ConvUp(1024, 256)

        self.convset_c1 = ConvSet(768, 256)
        self.outconv_c1 = OutConv(256, out_nums)
        self.convup_c10 = ConvUp(256, 128)

        self.convset_c0 = ConvSet(384, 128)
        self.outconv_c0 = OutConv(128, out_nums)

    def forward(self, x):
        features = self.backbone(x)

        C2 = self.convset_c2(features[2])
        C1 = self.convset_c1(torch.cat([features[1], self.convup_c21(C2)], dim=1))
        C0 = self.convset_c0(torch.cat([features[0], self.convup_c10(C1)], dim=1))

        # [52*52, 26*26, 13*13]
        C0 = self.outconv_c0(C0).permute(0, 2, 3, 1)
        C1 = self.outconv_c1(C1).permute(0, 2, 3, 1)
        C2 = self.outconv_c2(C2).permute(0, 2, 3, 1)

        C0 = C0.reshape(C0.shape[0], C0.shape[1], C0.shape[2], self.num_boxes, -1)
        C1 = C1.reshape(C1.shape[0], C1.shape[1], C1.shape[2], self.num_boxes, -1)
        C2 = C2.reshape(C2.shape[0], C2.shape[1], C2.shape[2], self.num_boxes, -1)

        return [C0, C1, C2]


def build_detector(cfg):
    return Detector(
        cfg,
        num_boxes=cfg["MODEL"]["NUM_BOXES"],
        num_classes=cfg["MODEL"]["NUM_CLASSES"],
    )
