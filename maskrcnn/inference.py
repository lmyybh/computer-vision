import os
from PIL import Image

import cv2
import toml
import torch


from maskrcnn.models.model import MaskRCNN
from maskrcnn.data.transforms import Compose, Resize, ToTensor, Normalize
from maskrcnn.data.dataloader import to_image_list


def build_model(cfg, MODEL_PATH, device):
    model = MaskRCNN(cfg)
    model.load_state_dict(torch.load(MODEL_PATH))
    model = model.eval()
    model = model.to(device)
    return model


def preprocess_images(img_paths, cfg, device):
    min_size = cfg["INPUT"]["MIN_SIZE_TEST"]
    max_size = cfg["INPUT"]["MAX_SIZE_TEST"]

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

    imgs = []
    for path in img_paths:
        img = Image.open(path).convert("RGB")
        img, _ = transforms_(img)
        imgs.append(img)

    imgs = to_image_list(imgs, cfg["DATA"]["SIZE_DIVISIBLE"])
    imgs = imgs.to(device)

    return imgs


def mark_images(img_paths, boxlists, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for img_path, boxlist in zip(img_paths, boxlists):
        _, img_name = os.path.split(img_path)
        img = cv2.imread(img_path)
        for box in boxlist.bbox:
            box = box.detach().cpu().numpy().astype(int)
            img = cv2.rectangle(img, box[:2], box[2:], (0, 0, 255), 2)
        marked_img_path = os.path.join(out_dir, f"marked_{img_name}")
        cv2.imwrite(marked_img_path, img)
        print(f"write {marked_img_path}")


# config
cfg = toml.load(
    "/home/cgl/projects/computer-vision/maskrcnn/maskrcnn/configs/config.toml"
)
MODEL_PATH = "/data1/cgl/tasks/coco_box/2022-09-13-17-20-56/checkpoints/epoch_12.pt"

device = torch.device("cuda:0")
model = build_model(cfg, MODEL_PATH, device)


img_paths = [
    "test.jpg",
    "/data1/cgl/dataset/coco2017/val2017/000000083113.jpg",
    "/data1/cgl/dataset/coco2017/val2017/000000083172.jpg",
    "/data1/cgl/dataset/coco2017/val2017/000000083531.jpg",
    "/data1/cgl/dataset/coco2017/val2017/000000083540.jpg",
    "/data1/cgl/dataset/coco2017/val2017/000000084031.jpg",
]
imgs = preprocess_images(img_paths, cfg, device)

detections = model(imgs)["detections"]

mark_images(
    img_paths, [detection.get_field("bboxes") for detection in detections], "./output"
)
