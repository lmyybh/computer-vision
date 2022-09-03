import torch

# from torchvision.ops import nms


class BoxList(object):
    def __init__(self, info, img_size, mode="xyxy"):
        device = info.device if isinstance(info, torch.Tensor) else torch.device("cpu")
        info = torch.as_tensor(info, dtype=torch.float32, device=device)
        assert (
            info.ndim == 2 and info.size(-1) == 5
        ), f"info (shape={info.shape}) 必须为 [n, 5] 的张量"
        assert mode in ("xyxy", "xywh"), f'mode={mode} 必须是 "xyxy" 或者 "xywh"'

        self.info = info
        self.size = img_size  # (w, h)
        self.mode = mode

    @property
    def bbox(self):
        return self.info[:, :4]

    @property
    def labels(self):
        return self.info[:, -1]

    def _split_into_xyxy(self):
        assert self.mode in ("xyxy", "xywh"), f'mode={self.mode} 必须是 "xyxy" 或者 "xywh"'
        if self.mode == "xyxy":
            return self.bbox.split(1, dim=1)
        else:
            xmin, ymin, w, h = self.bbox.split(1, dim=1)
            xmax = xmin + w.clamp(min=0)
            ymax = ymin + h.clamp(min=0)
            return xmin, ymin, xmax, ymax

    def convert(self, mode):
        assert mode in ("xyxy", "xywh"), f'mode={mode} 必须是 "xyxy" 或者 "xywh"'
        if self.mode == mode:
            return self
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if mode == "xyxy":
            bbox = torch.cat((xmin, ymin, xmax, ymax), dim=1)
        else:
            bbox = torch.cat((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)
        info = torch.hstack((bbox, self.labels.reshape(-1, 1)))
        return BoxList(info, self.size, mode)

    def resize(self, size):
        ratio_width, ratio_height = size[0] / self.size[0], size[1] / self.size[1]
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        xmin *= ratio_width
        xmax *= ratio_width
        ymin *= ratio_height
        ymax *= ratio_height
        bbox = torch.cat((xmin, ymin, xmax, ymax), dim=1)
        info = torch.hstack((bbox, self.labels.reshape(-1, 1)))
        boxlist = BoxList(info, size, mode="xyxy")

        return boxlist.convert(mode=self.mode)

    def area(self):
        box = self.bbox
        if self.mode == "xyxy":
            area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        else:
            area = box[:, 2] * box[:, 3]
        return area

    def to(self, device):
        return BoxList(self.info.to(device), self.size, self.mode)

    def clip_to_image(self, remove_empty=True):
        boxlist = self.convert("xyxy")
        bbox = boxlist.bbox
        bbox[:, 0].clamp_(min=0, max=self.size[0] - 1)
        bbox[:, 1].clamp_(min=0, max=self.size[1] - 1)
        bbox[:, 2].clamp_(min=0, max=self.size[0] - 1)
        bbox[:, 3].clamp_(min=0, max=self.size[1] - 1)

        info = torch.hstack((bbox, boxlist.labels.reshape(-1, 1)))
        boxlist = BoxList(info, boxlist.size, boxlist.mode)

        if remove_empty:
            bbox = boxlist.bbox
            keep = (bbox[:, 3] > bbox[:, 1]) & (bbox[:, 2] > bbox[:, 0])
            return boxlist[keep].convert(self.mode)

        return boxlist.convert(self.mode)

    def remove_small_boxes(self, min_size):
        boxlist = self.convert("xywh")
        _, _, w, h = boxlist.bbox.unbind(dim=1)
        keep = ((w > min_size) & (h > min_size)).nonzero().squeeze(1)

        return self[keep]

    # def boxlist_nms(self, iou_threshold, post_nms_top_n=-1):
    #     if iou_threshold <= 0:
    #         return self

    #     idxs = nms(self.convert("xyxy").bbox, self.scores, iou_threshold)

    #     if post_nms_top_n > 0:
    #         idxs = idxs[:post_nms_top_n]

    #     return self[idxs]

    def __len__(self):
        return self.info.shape[0]

    def __getitem__(self, index):
        return BoxList(self.info[index], self.size, self.mode)
