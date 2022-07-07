import torchvision.transforms.functional as TF


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Resize(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size
    
    def get_size(self, image_size):
        w, h = image_size
        min_size, max_size = self.min_size, self.max_size

        # 获取图片长短边
        max_origin_size = float(max(w, h))
        min_origin_size = float(min(w, h))

        # 如果原图相比设定大小更加细长，则长边为max_size，短边按比例调整
        if max_origin_size / min_origin_size > max_size / min_size:
            min_size = int(round(max_size * min_origin_size / max_origin_size))
        
        # 此时已经确定min_size，即可按比例确定任一图片的max_size
        if w < h:
            ow = min_size
            oh = int(ow * h / w)
        else:
            oh = min_size
            ow = int(oh * w / h)
        return (ow, oh)


    def __call__(self, image, target):
        ow, oh = self.get_size(image.size)
        image = TF.resize(image, (oh, ow))
        target = target.resize((ow, oh))
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = TF.to_tensor(image)
        return image, target
