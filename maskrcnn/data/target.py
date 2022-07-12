import torch
import torchvision.transforms.functional as TF

class BoxList(object):
    def __init__(self, bbox, img_size, mode='xyxy'):
        device = bbox.device if isinstance(bbox, torch.Tensor) else torch.device('cpu')
        bbox = torch.as_tensor(bbox, dtype=torch.float32, device=device)
        assert bbox.ndim == 2 and bbox.size(-1) == 4, f'bbox (shape={bbox.shape}) 必须为 [n, 4] 的张量'
        assert mode in ('xyxy', 'xywh'), f'mode={mode} 必须是 "xyxy" 或者 "xywh"'

        self.bbox = bbox
        self.size = img_size #(w, h)
        self.mode = mode
    
    def _split_into_xyxy(self):
        assert self.mode in ('xyxy', 'xywh'), f'mode={self.mode} 必须是 "xyxy" 或者 "xywh"'
        if self.mode == 'xyxy':
            return self.bbox.split(1, dim=1)
        else:
            xmin, ymin, w, h = self.bbox.split(1, dim=1)
            xmax = xmin + (w - 1).clamp(min=0)
            ymax = ymin + (h - 1).clamp(min=0)
            return xmin, ymin, xmax, ymax

    def convert(self, mode):
        assert mode in ('xyxy', 'xywh'), f'mode={mode} 必须是 "xyxy" 或者 "xywh"'
        if self.mode == mode:
            return self
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if mode == 'xyxy':
            bbox = torch.cat((xmin, ymin, xmax, ymax), dim=1)
        else:
            bbox = torch.cat(
                (xmin, ymin, xmax - xmin + 1, ymax - ymin + 1), dim=1
            )
        return BoxList(bbox, self.size, mode)
    
    def resize(self, size):
        ratio_width, ratio_height = size[0] / self.size[0], size[1] / self.size[1]
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        xmin *= ratio_width
        xmax *= ratio_width
        ymin *= ratio_height
        ymax *= ratio_height
        bbox = torch.cat((xmin, ymin, xmax, ymax), dim=1)
        bbox = BoxList(bbox, size, mode='xyxy')

        return bbox.convert(mode=self.mode)
    
    def to(self, device):
        return BoxList(self.bbox.to(device), self.size, self.mode)

class Mask(object):
    def __init__(self, mask):
        device = mask.device if isinstance(mask, torch.Tensor) else torch.device('cpu')
        self.mask = torch.as_tensor(mask, dtype=torch.float32, device=device)
    
    def resize(self, size):
        return Mask(TF.resize(self.mask, size[::-1])) # size = (ow, oh)，需要以 (oh, ow) 形式输入

    def to(self, device):
        return Mask(self.mask.to(device))


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
