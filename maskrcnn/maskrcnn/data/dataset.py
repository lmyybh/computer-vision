import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pycocotools.coco import COCO

from .target import Target, BoxList, Mask
from .transforms import Compose, Resize, ToTensor, Normalize, ColorJitter


class CocoDataset(Dataset):
    def __init__(self, image_dir, ann_file, transforms):
        self.image_dir = image_dir
        self.coco = COCO(ann_file)
        self.img_ids = [
            img_id
            for img_id in sorted(list(self.coco.imgs.keys()))
            if self.has_valid_annotation(img_id)
        ]
        self.continuous_labels = {v: i + 1 for i, v in enumerate(self.coco.getCatIds())}
        self.transforms = transforms

    def __len__(self):
        return len(self.img_ids)

    def get_image(self, img_id):
        filename = self.coco.loadImgs(img_id)[0]["file_name"]
        filepath = os.path.join(self.image_dir, filename)
        return Image.open(filepath).convert("RGB")

    def has_valid_annotation(self, img_id):
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        # 不存在标注
        if len(anns) <= 0:
            return False

        # 标注区域为空
        if all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anns):
            return False

        return True

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))

        image = self.get_image(img_id)

        labels, boxes, masks = [], [], []
        for ann in anns:
            labels.append(ann["category_id"])
            boxes.append(ann["bbox"])
            masks.append(self.coco.annToMask(ann))
        # to Tensor
        labels = [self.continuous_labels[label] for label in labels]
        labels = torch.tensor(labels).float()
        boxes = torch.tensor(np.array(boxes)).float()
        masks = torch.tensor(np.stack(masks, axis=0)).float()

        target = Target()
        target.add_field("labels", labels)
        target.add_field(
            "bboxes", BoxList(boxes, image.size, mode="xywh").convert("xyxy")
        )
        target.add_field("masks", Mask(masks))

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


def build_dataset(cfg, is_train=True):
    if is_train:
        min_size = cfg["INPUT"]["MIN_SIZE_TRAIN"]
        max_size = cfg["INPUT"]["MAX_SIZE_TRAIN"]
        image_dir = cfg["DATA"]["IMAGE_DIR_TRAIN"]
        ann_file = cfg["DATA"]["ANN_FILE_TRAIN"]
    else:
        min_size = cfg["INPUT"]["MIN_SIZE_TEST"]
        max_size = cfg["INPUT"]["MAX_SIZE_TEST"]
        image_dir = cfg["DATA"]["IMAGE_DIR_TEST"]
        ann_file = cfg["DATA"]["ANN_FILE_TEST"]

    normalize_transform = Normalize(
        mean=cfg["INPUT"]["PIXEL_MEAN"],
        std=cfg["INPUT"]["PIXEL_STD"],
    )

    transforms_ = Compose(
        [
            Resize(min_size, max_size),
            ToTensor(),
            normalize_transform,
        ]
    )

    return CocoDataset(
        image_dir=image_dir,
        ann_file=ann_file,
        transforms=transforms_,
    )
