import torch.nn as nn
import torch.nn.functional as F

from maskrcnn.models.poolers import Pooler


def make_fc(dim_in, hidden_dim, use_gn=False):
    fc = nn.Linear(dim_in, hidden_dim, bias=(not use_gn))
    nn.init.kaiming_uniform_(fc.weight, a=1)

    if use_gn:
        return nn.Sequential(fc, F.group_norm(hidden_dim))

    nn.init.constant_(fc.bias, 0)
    return fc


class FPN2MLPFeatureExtractor(nn.Module):
    def __init__(
        self,
        in_channels,
        resolution,
        scales,
        sampling_ratio,
        representation_size,
        use_gn=False,
    ):
        super(FPN2MLPFeatureExtractor, self).__init__()

        self.pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        input_size = in_channels * resolution**2
        self.fc6 = make_fc(input_size, representation_size, use_gn)
        self.fc7 = make_fc(representation_size, representation_size, use_gn)
        self.out_channels = representation_size

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x


def make_roi_box_feature_extractor(cfg, in_channels):
    return FPN2MLPFeatureExtractor(
        in_channels,
        resolution=cfg["MODEL"]["ROI_BOX_HEAD"]["POOLER_RESOLUTION"],
        scales=cfg["MODEL"]["ROI_BOX_HEAD"]["POOLER_SCALES"],
        sampling_ratio=cfg["MODEL"]["ROI_BOX_HEAD"]["POOLER_SAMPLING_RATIO"],
        representation_size=cfg["MODEL"]["ROI_BOX_HEAD"]["MLP_HEAD_DIM"],
        use_gn=cfg["MODEL"]["ROI_BOX_HEAD"]["USE_GN"],
    )
