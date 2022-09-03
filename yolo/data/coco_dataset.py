import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from pycocotools.coco import COCO

from .boxlist import BoxList
from .transforms import Compose, Resize


class CocoDataset(Dataset):
    def __init__(self, image_dir, ann_file, transforms=None):
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

    def get_image(self, img_id, pad_to_square=False):
        filename = self.coco.loadImgs(img_id)[0]["file_name"]
        filepath = os.path.join(self.image_dir, filename)
        img = Image.open(filepath).convert("RGB")
        img = to_tensor(img)
        if pad_to_square:
            max_wh = max(img.shape[1:])
            size = (3, max_wh, max_wh)
            padded_img = img.new_zeros(size)
            padded_img[: img.shape[0], : img.shape[1], : img.shape[2]] = img
            return padded_img
        return img

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

        image = self.get_image(img_id, pad_to_square=True)

        labels, boxes = [], []
        for ann in anns:
            labels.append(ann["category_id"])
            boxes.append(ann["bbox"])

        # to Tensor
        labels = [self.continuous_labels[label] for label in labels]
        labels = torch.tensor(labels).float()
        boxes = torch.tensor(np.array(boxes)).float()
        info = torch.hstack((boxes, labels.reshape(-1, 1)))
        boxlist = BoxList(info, image.shape[1:], "xywh").convert("xyxy")

        if self.transforms is not None:
            image, boxlist = self.transforms(image, boxlist)

        return image, boxlist


def build_dataset(cfg, is_train=True):
    if is_train:
        image_dir = cfg["DATA"]["IMAGE_DIR_TRAIN"]
        ann_file = cfg["DATA"]["ANN_FILE_TRAIN"]
    else:
        image_dir = cfg["DATA"]["IMAGE_DIR_VAL"]
        ann_file = cfg["DATA"]["ANN_FILE_VAL"]

    transforms = Compose([Resize(size=(448, 448))])

    return CocoDataset(image_dir, ann_file, transforms=transforms)
