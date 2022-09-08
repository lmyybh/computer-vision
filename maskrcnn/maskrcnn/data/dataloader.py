from math import ceil
import torch
from torch.utils.data import DataLoader

from .dataset import build_dataset


class ImageList(object):
    def __init__(self, tensors, image_sizes):
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, *args, **kwargs):
        tensors = self.tensors.to(*args, **kwargs)
        return ImageList(tensors, self.image_sizes)


def to_image_list(images, size_divisible):
    max_size = tuple(
        max(s) for s in zip(*[img.shape for img in images])
    )  # (max_channle, max_width, max_height)

    # 保证下采样的过程中，图片尺寸始终可以整除
    if size_divisible > 0:
        max_size = list(max_size)
        max_size[1] = int(ceil(max_size[1] / size_divisible) * size_divisible)
        max_size[2] = int(ceil(max_size[2] / size_divisible) * size_divisible)
        max_size = tuple(max_size)

    batch_size = len(images)
    tensors = torch.zeros(*(batch_size, *max_size))
    for img, pad_img in zip(images, tensors):
        # 图片右下补零到指定尺寸，这种方式不会影响 bbox 坐标
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

    # 记录图片原始尺寸
    image_sizes = [img.shape[1:] for img in images]

    return ImageList(tensors, image_sizes)


class BatchCollator(object):
    def __init__(self, size_divisible):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        images, targets = list(zip(*batch))
        images = to_image_list(images, self.size_divisible)
        return images, targets


def build_dataloader(cfg, is_train=True):
    shuffle = is_train
    batch_size = (
        cfg["DATA"]["BATCH_SIZE_TRAIN"] if is_train else cfg["DATA"]["BATCH_SIZE_TEST"]
    )
    return DataLoader(
        build_dataset(cfg, is_train=is_train),
        batch_size=batch_size,
        collate_fn=BatchCollator(cfg["DATA"]["SIZE_DIVISIBLE"]),
        shuffle=shuffle,
        num_workers=cfg["DATA"]["NUM_WORKERS"],
    )
