import torchvision.transforms.functional as F


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, boxmgr=None):
        for t in self.transforms:
            image, boxmgr = t(image, boxmgr)
        return image, boxmgr


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, boxmgr=None):
        image = F.resize(image, self.size)
        if boxmgr:
            boxmgr = boxmgr.resize(list(self.size)[::-1])
        return image, boxmgr


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, boxmgr=None):
        image = F.normalize(image, mean=self.mean, std=self.std)

        return image, boxmgr
