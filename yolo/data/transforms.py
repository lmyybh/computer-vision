import torchvision.transforms.functional as F


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, boxlist=None):
        for t in self.transforms:
            image, boxlist = t(image, boxlist)
        return image, boxlist


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, boxlist=None):
        image = F.resize(image, self.size)
        if boxlist:
            boxlist = boxlist.resize(list(self.size)[::-1])
        return image, boxlist
