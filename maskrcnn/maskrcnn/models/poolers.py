import torch
import torch.nn as nn
from torchvision.ops import RoIAlign
from .rpn.utils import cat


class LevelMapper:
    def __init__(self, k_min, k_max, canonical_scale=224, canonical_level=4, eps=1e-6):
        self.k_min = k_min
        self.k_max = k_max
        self.s0 = canonical_scale
        self.lv10 = canonical_level
        self.eps = eps

    def __call__(self, boxlists):
        s = torch.sqrt(cat([boxlist.area() for boxlist in boxlists]))

        target_lvls = torch.floor(self.lv10 + torch.log2(s / self.s0 + self.eps))
        target_lvls = torch.clamp(target_lvls, min=self.k_min, max=self.k_max)
        return target_lvls.to(torch.int64) - self.k_min


class Pooler(nn.Module):
    def __init__(self, output_size, scales, sampling_ratio):
        super().__init__()

        poolers = []
        for scale in scales:
            poolers.append(RoIAlign(output_size, scale, sampling_ratio))
        self.poolers = nn.ModuleList(poolers)
        self.output_size = output_size

        lvl_min = -torch.log2(torch.tensor(scales[0], dtype=torch.float32)).item()
        lvl_max = -torch.log2(torch.tensor(scales[-1], dtype=torch.float32)).item()
        self.map_levels = LevelMapper(lvl_min, lvl_max)

    def convert_to_roi_format(self, boxes):
        concat_boxes = cat([b.bbox for b in boxes], dim=0)
        device, dtype = concat_boxes.device, concat_boxes.dtype
        ids = cat(
            [
                torch.full((len(b), 1), i, device=device, dtype=dtype)
                for i, b in enumerate(boxes)
            ],
            dim=0,
        )
        return torch.cat([ids, concat_boxes], dim=1)

    def forward(self, x, boxes):
        num_levels = len(self.poolers)
        rois = self.convert_to_roi_format(boxes)

        if num_levels == 1:
            return self.poolers[0](x[0], rois)

        levels = self.map_levels(boxes)

        num_rois = len(rois)
        num_channels = x[0].shape[1]
        output_size = self.output_size[0]

        device, dtype = x[0].device, x[0].dtype
        result = torch.zeros(
            (num_rois, num_channels, output_size, output_size),
            dtype=dtype,
            device=device,
        )

        for level, (per_level_feature, pooler) in enumerate(zip(x, self.poolers)):
            idx_in_level = torch.nonzero(levels == level).squeeze(1)
            rois_per_level = rois[idx_in_level]
            result[idx_in_level] = pooler(per_level_feature, rois_per_level).to(dtype)

        return result
