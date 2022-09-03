from PIL import Image

import cv2
import toml
import numpy as np
import torch
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt

from maskrcnn.data import build_dataset, build_dataloader
from maskrcnn.data.target import cat_boxlists
from maskrcnn.models.backbone import build_backbone
from maskrcnn.models.rpn import build_rpn
from maskrcnn.models.model import MaskRCNN


def PIL2CV(image):
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def CV2PIL(image):
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def de_normalized(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean)
    if not isinstance(std, torch.Tensor):
        std = torch.tensor(std)
    mean = mean.unsqueeze(1).unsqueeze(1)
    std = std.unsqueeze(1).unsqueeze(1)
    return tensor * std + mean


def show(images, boxlists, figsize=(10, 10)):
    images = images.to("cpu")
    TP = ToPILImage()
    for image, size, boxlist in zip(images.tensors, images.image_sizes, boxlists):
        image = image[:, : size[0], : size[1]]
        image = PIL2CV(TP(de_normalized(image)))

        for box in boxlist.bbox:
            box = box.cpu().numpy().astype(int)
            image = cv2.rectangle(image, box[:2], box[2:], (0, 255, 0), 1)

        image = CV2PIL(image)

        plt.figure(figsize=figsize)
        plt.imshow(image)
        plt.axis("off")
        plt.show()


cfg = toml.load(
    "/home/cgl/projects/computer-vision/maskrcnn/maskrcnn/configs/config.toml"
)
device = torch.device("cuda:0")

dataset = build_dataset(cfg, is_train=True)
dataloader = build_dataloader(dataset, cfg, is_train=True)

for i, (images, targets) in enumerate(dataloader):
    images = images.to(device)
    targets = [target.to(device) for target in targets]
    break

model = MaskRCNN(cfg).to(device)

model(images, targets)
