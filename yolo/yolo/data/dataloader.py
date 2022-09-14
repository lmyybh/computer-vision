import torch
from torch.utils.data import DataLoader

from .coco_dataset import build_dataset


def batch_collator(batch):
    images, boxmgrs = list(zip(*batch))
    images = torch.stack(images, dim=0)
    return images, boxmgrs


def build_dataloader(cfg, is_train=True):
    dataset = build_dataset(cfg, is_train=is_train)

    batch_size = (
        cfg["DATA"]["BATCH_SIZE_TRAIN"] if is_train else cfg["DATA"]["BATCH_SIZE_VAL"]
    )

    return DataLoader(
        dataset, batch_size, shuffle=is_train, num_workers=10, collate_fn=batch_collator
    )
