import torch
from torch.utils.data import DataLoader

from .coco_dataset import build_dataset


class Target:
    def __init__(self, boxlists):
        self.boxlists = boxlists

    def to(self, device):
        self.boxlists = [b.to(device) for b in self.boxlists]
        return self

    def __getitem__(self, idx):
        return self.boxlists[idx]


def batch_collator(batch):
    images, boxlists = list(zip(*batch))
    images = torch.stack(images, dim=0)
    targets = Target(boxlists)
    return images, targets


def build_dataloader(cfg, is_train=True):
    dataset = build_dataset(cfg, is_train=is_train)

    batch_size = (
        cfg["DATA"]["BATCH_SIZE_TRAIN"] if is_train else cfg["DATA"]["BATCH_SIZE_VAL"]
    )

    return DataLoader(
        dataset, batch_size, shuffle=is_train, num_workers=10, collate_fn=batch_collator
    )
