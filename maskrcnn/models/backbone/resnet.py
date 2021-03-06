import torch.nn as nn
from torchvision.models import resnet101


class ResNet101(nn.Module):
    def __init__(self):
        super(ResNet101, self).__init__()
        self.model = resnet101(pretrained=True, progress=True)

    def forward(self, x):
        outputs = []

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        outputs.append(x)  # C2: [N, 256, w/4, h/4]

        x = self.model.layer2(x)
        outputs.append(x)  # C3: [N, 512, w/8, h/8]

        x = self.model.layer3(x)
        outputs.append(x)  # C4: [N, 1024, w/16, h/16]

        x = self.model.layer4(x)
        outputs.append(x)  # C5: [N, 2048, w/32, h/32]

        return outputs
