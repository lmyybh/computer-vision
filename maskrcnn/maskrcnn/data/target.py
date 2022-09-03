import torch
import torchvision.transforms.functional as TF
from torchvision.ops import nms


class BoxList(object):
    def __init__(self, bbox, img_size, scores=None, mode="xyxy"):
        device = bbox.device if isinstance(bbox, torch.Tensor) else torch.device("cpu")
        bbox = torch.as_tensor(bbox, dtype=torch.float32, device=device)
        assert (
            bbox.ndim == 2 and bbox.size(-1) == 4
        ), f"bbox (shape={bbox.shape}) 必须为 [n, 4] 的张量"
        assert mode in ("xyxy", "xywh"), f'mode={mode} 必须是 "xyxy" 或者 "xywh"'

        self.bbox = bbox
        self.size = img_size  # (w, h)
        self.scores = scores
        self.mode = mode

    def _split_into_xyxy(self):
        assert self.mode in ("xyxy", "xywh"), f'mode={self.mode} 必须是 "xyxy" 或者 "xywh"'
        if self.mode == "xyxy":
            return self.bbox.split(1, dim=1)
        else:
            xmin, ymin, w, h = self.bbox.split(1, dim=1)
            xmax = xmin + (w - 1).clamp(min=0)
            ymax = ymin + (h - 1).clamp(min=0)
            return xmin, ymin, xmax, ymax

    def convert(self, mode):
        assert mode in ("xyxy", "xywh"), f'mode={mode} 必须是 "xyxy" 或者 "xywh"'
        if self.mode == mode:
            return self
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if mode == "xyxy":
            bbox = torch.cat((xmin, ymin, xmax, ymax), dim=1)
        else:
            bbox = torch.cat((xmin, ymin, xmax - xmin + 1, ymax - ymin + 1), dim=1)
        return BoxList(bbox, self.size, self.scores, mode)

    def resize(self, size):
        ratio_width, ratio_height = size[0] / self.size[0], size[1] / self.size[1]
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        xmin *= ratio_width
        xmax *= ratio_width
        ymin *= ratio_height
        ymax *= ratio_height
        bbox = torch.cat((xmin, ymin, xmax, ymax), dim=1)
        bbox = BoxList(bbox, size, mode="xyxy")

        return bbox.convert(mode=self.mode)

    def area(self):
        box = self.bbox
        if self.mode == "xyxy":
            area = (box[:, 2] - box[:, 0] + 1) * (box[:, 3] - box[:, 1] + 1)
        else:
            area = box[:, 2] * box[:, 3]
        return area

    def to(self, device):
        return BoxList(self.bbox.to(device), self.size, self.scores, self.mode)

    def clip_to_image(self, remove_empty=True):
        assert self.mode == "xyxy"
        self.bbox[:, 0].clamp_(min=0, max=self.size[0] - 1)
        self.bbox[:, 1].clamp_(min=0, max=self.size[1] - 1)
        self.bbox[:, 2].clamp_(min=0, max=self.size[0] - 1)
        self.bbox[:, 3].clamp_(min=0, max=self.size[1] - 1)

        if remove_empty:
            box = self.bbox
            keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])
            return self[keep]

        return self

    def remove_small_boxes(self, min_size):
        boxlist = self.convert("xywh")
        _, _, w, h = boxlist.bbox.unbind(dim=1)
        keep = ((w > min_size) & (h > min_size)).nonzero().squeeze(1)

        return self[keep]

    def boxlist_nms(self, iou_threshold, post_nms_top_n=-1):
        if iou_threshold <= 0:
            return self

        idxs = nms(self.convert("xyxy").bbox, self.scores, iou_threshold)

        if post_nms_top_n > 0:
            idxs = idxs[:post_nms_top_n]

        return self[idxs]

    def __len__(self):
        return self.bbox.shape[0]

    def __getitem__(self, index):
        scores = self.scores[index] if self.scores is not None else None
        return BoxList(self.bbox[index], self.size, scores, self.mode)


def cat_boxlists(boxlists):
    assert isinstance(boxlists, (tuple, list))
    assert len(boxlists) > 0

    if len(boxlists) == 1:
        return boxlists[0]

    assert all(isinstance(box, BoxList) for box in boxlists)

    image_size = boxlists[0].size
    assert all(box.size == image_size for box in boxlists)

    mode = boxlists[0].mode
    assert all(box.mode == mode for box in boxlists)

    scores = boxlists[0].scores
    assert all(isinstance(box.scores, type(scores)) for box in boxlists)

    cat_bbox = torch.cat([box.bbox for box in boxlists], dim=0)
    cat_scores = (
        torch.cat([box.scores for box in boxlists], dim=0)
        if scores is not None
        else None
    )

    return BoxList(cat_bbox, image_size, cat_scores, mode)


def boxlist_iou(boxlist1, boxlist2):
    assert boxlist1.size == boxlist2.size

    boxlist1 = boxlist1.convert("xyxy")
    boxlist2 = boxlist2.convert("xyxy")

    N, M = len(boxlist1), len(boxlist2)
    area1, area2 = boxlist1.area(), boxlist2.area()
    box1, box2 = boxlist1.bbox, boxlist2.bbox

    lt = torch.max(box1[:, None, :2], box2[:, :2])
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])

    wh = (rb - lt + 1).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    return inter / (area1[:, None] + area2 - inter)


class Mask(object):
    def __init__(self, mask):
        device = mask.device if isinstance(mask, torch.Tensor) else torch.device("cpu")
        self.mask = torch.as_tensor(mask, dtype=torch.float32, device=device)

    def resize(self, size):
        return Mask(
            TF.resize(self.mask, size[::-1])
        )  # size = (ow, oh)，需要以 (oh, ow) 形式输入

    def to(self, device):
        return Mask(self.mask.to(device))

    def __getitem__(self, index):
        return Mask(self.mask[index])


class Target(object):
    def __init__(self):
        self.fields = {}

    def add_field(self, key, value):
        self.fields[key] = value

    def get_field(self, key):
        return self.fields[key]

    def resize(self, size):
        resized_target = Target()
        for key in self.fields.keys():
            if isinstance(self.get_field(key), (BoxList, Mask)):
                resized_target.add_field(key, self.get_field(key).resize(size))
            else:
                resized_target.add_field(key, self.get_field(key))
        return resized_target

    def to(self, device):
        new_target = Target()
        for key in self.fields.keys():
            if isinstance(self.get_field(key), (torch.Tensor, BoxList, Mask)):
                new_target.add_field(key, self.get_field(key).to(device))
            else:
                new_target.add_field(key, self.get_field(key))
        return new_target

    def __getitem__(self, index):
        target = Target()
        for k, v in self.fields.items():
            target.add_field(k, v[index])
        return target
