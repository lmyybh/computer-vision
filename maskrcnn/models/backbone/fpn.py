import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
    conv = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    nn.init.kaiming_uniform_(conv.weight, a=1)
    return conv


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        self.inner_blocks = nn.ModuleList([])
        self.layer_blocks = nn.ModuleList([])

        for in_channels in in_channels_list:
            self.inner_blocks.append(
                conv_block(in_channels, out_channels, 1, stride=1, padding=0)
            )

            self.layer_blocks.append(
                conv_block(out_channels, out_channels, 3, stride=1, padding=1)
            )

    def forward(self, x):
        outputs = []

        last_inner = self.inner_blocks[-1](x[-1])
        P5 = self.layer_blocks[-1](last_inner)
        P6 = F.max_pool2d(P5, kernel_size=3, stride=2, padding=1)
        outputs += [P6, P5]

        for feature, inner, layer in zip(
            x[-2::-1], self.inner_blocks[-2::-1], self.layer_blocks[-2::-1]
        ):
            last_inner = inner(feature) + F.interpolate(
                last_inner, scale_factor=2, mode="nearest"
            )

            outputs.append(layer(last_inner))

        # 计算顺序是倒序，所以需要反序输出
        return outputs[::-1]
