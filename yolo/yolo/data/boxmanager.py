import torch

# from torchvision.ops import nms


class BoxManager(object):
    def __init__(self, bbox, img_size=None, labels=None, scores=None, mode="xyxy"):
        device = bbox.device if isinstance(bbox, torch.Tensor) else torch.device("cpu")
        self.bbox = torch.as_tensor(bbox, dtype=torch.float32, device=device)
        self.img_size = img_size
        self.labels = labels
        self.scores = scores
        self.mode = mode

    def _split_into_xyxy(self):
        assert self.mode in (
            "xyxy",
            "xywh",
            "cxywh",
        ), f'mode={self.mode} 必须是 "xyxy", "xywh" 或者 "cxywh"'
        if self.mode == "xyxy":
            return self.bbox.split(1, dim=1)
        elif self.mode == "xywh":
            xmin, ymin, w, h = self.bbox.split(1, dim=1)
            xmax = xmin + w.clamp(min=0)
            ymax = ymin + h.clamp(min=0)
            return xmin, ymin, xmax, ymax
        else:
            ctx, cty, w, h = self.bbox.split(1, dim=1)
            xmin = ctx - (w / 2).clamp(min=0)
            xmax = ctx + (w / 2).clamp(min=0)
            ymin = cty - (h / 2).clamp(min=0)
            ymax = cty + (h / 2).clamp(min=0)
            return xmin, ymin, xmax, ymax

    def convert(self, mode):
        assert mode in (
            "xyxy",
            "xywh",
            "cxywh",
        ), f'mode={mode} 必须是 "xyxy", "xywh 或者 "cxywh"'
        if self.mode == mode:
            return self
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if mode == "xyxy":
            bbox = torch.cat((xmin, ymin, xmax, ymax), dim=1)
        elif mode == "xywh":
            bbox = torch.cat((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)
        elif mode == "cxywh":
            bbox = torch.cat(
                ((xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin), dim=1
            )

        return BoxManager(
            bbox,
            img_size=self.img_size,
            labels=self.labels,
            scores=self.scores,
            mode=mode,
        )

    def resize(self, size):
        ratio_width, ratio_height = (
            size[0] / self.img_size[0],
            size[1] / self.img_size[1],
        )
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        xmin *= ratio_width
        xmax *= ratio_width
        ymin *= ratio_height
        ymax *= ratio_height
        bbox = torch.cat((xmin, ymin, xmax, ymax), dim=1)

        boxmgr = BoxManager(bbox, self.img_size, self.labels, self.scores, mode="xyxy")

        return boxmgr.convert(mode=self.mode)

    def area(self):
        box = self.bbox
        if self.mode == "xyxy":
            area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        else:
            area = box[:, 2] * box[:, 3]
        return area

    def to(self, device):
        bbox = self.bbox.to(device)
        labels = (
            self.labels.to(device)
            if isinstance(self.labels, torch.Tensor)
            else self.labels
        )
        scores = (
            self.scores.to(device)
            if isinstance(self.scores, torch.Tensor)
            else self.scores
        )

        return BoxManager(bbox, self.img_size, labels, scores, self.mode)

    def clip_to_image(self, remove_empty=True):
        boxmgr = self.convert("xyxy")

        bbox = torch.zeros_like(boxmgr.bbox)
        bbox[:, 0] = torch.clamp(boxmgr.bbox[:, 0], min=0, max=self.size[0] - 1)
        bbox[:, 1] = torch.clamp(boxmgr.bbox[:, 1], min=0, max=self.size[1] - 1)
        bbox[:, 2] = torch.clamp(boxmgr.bbox[:, 2], min=0, max=self.size[0] - 1)
        bbox[:, 3] = torch.clamp(boxmgr.bbox[:, 3], min=0, max=self.size[1] - 1)

        boxmgr = BoxManager(bbox, self.img_size, self.labels, self.scores, "xyxy")

        if remove_empty:
            bbox = boxmgr.bbox
            keep = (bbox[:, 3] > bbox[:, 1]) & (bbox[:, 2] > bbox[:, 0])
            return boxmgr[keep].convert(self.mode)

        return boxmgr.convert(self.mode)

    def remove_small_boxes(self, min_size):
        boxmgr = self.convert("xywh")
        _, _, w, h = boxmgr.bbox.unbind(dim=1)
        keep = ((w > min_size) & (h > min_size)).nonzero().squeeze(1)

        return self[keep]

    def __len__(self):
        return self.bbox.shape[0]

    def __getitem__(self, index):
        bbox = self.bbox[index]
        labels = (
            self.labels[index] if isinstance(self.labels, torch.Tensor) else self.labels
        )
        scores = (
            self.scores[index] if isinstance(self.scores, torch.Tensor) else self.scores
        )
        return BoxManager(bbox, self.img_size, labels, scores, self.mode)


def cat(datas, dim=-1):
    assert isinstance(datas, (tuple, list))
    assert len(datas) > 0

    if len(datas) == 1:
        return datas[0]

    data_type = type(datas[0])
    assert data_type in {type(None), torch.Tensor}
    assert all(isinstance(data, data_type) for data in data_type)

    if datas[0] is None:
        return None
    else:
        return torch.cat(datas, dim=dim)


def cat_boxmgrs(boxmgrs):
    assert isinstance(boxmgrs, (tuple, list))
    assert len(boxmgrs) > 0

    if len(boxmgrs) == 1:
        return boxmgrs[0]

    assert all(isinstance(box, BoxManager) for box in boxmgrs)

    image_size = boxmgrs[0].size
    assert all(box.size == image_size for box in boxmgrs)

    mode = boxmgrs[0].mode
    assert all(box.mode == mode for box in boxmgrs)

    bbox = cat([box.bbox for box in boxmgrs], dim=0)
    labels = cat([box.labels for box in boxmgrs])
    scores = cat([box.scores for box in boxmgrs])

    return BoxManager(bbox, image_size, labels, scores, mode)
