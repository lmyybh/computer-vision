import torch
import numpy as np

from maskrcnn.data.target import BoxList


class AnchorGenerator(torch.nn.Module):
    def __init__(
        self,
        strides=[4, 8, 16, 32, 64],
        sizes=[32, 64, 128, 256, 512],
        ratios=[0.5, 1, 2],
    ):
        super(AnchorGenerator, self).__init__()
        self.strides = strides
        # 每个尺寸特征图上的 anchors 组成的列表
        self.cell_anchors = generate_anchors(sizes, ratios)

    def num_anchors_per_location(self):
        return self.cell_anchors[0].shape[0]

    def grid_anchors(self, grid_sizes):
        # grid 即特征图，grid_sizes 就是特征图的尺寸
        # 例如：P2-P6 [[h/4, w/4], [h/8, w/8], [h/16, w/16], [h/32, w/32], [h/64, w/64]]
        anchors = []
        for size, stride, base_anchors in zip(
            grid_sizes, self.strides, self.cell_anchors
        ):
            h, w = size

            # 获取 grid (特征图) 上每个像素点对应在原图上的坐标
            shifts_x = torch.arange(0, w * stride, step=stride, dtype=torch.float32)
            shifts_y = torch.arange(0, h * stride, step=stride, dtype=torch.float32)
            # 注意对于图片来讲, y是行, x是列
            shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            shifts_x = shifts_x.flatten()
            shifts_y = shifts_y.flatten()

            # 组合成 xyxy 形式
            shifts = torch.stack((shifts_x, shifts_y, shifts_x, shifts_y), dim=1)

            # 平移 base anchors 到中心点
            anchors.append(
                (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
            )
        return anchors

    def forward(self, image_list, features):
        device = features[0].device
        grid_sizes = [f.shape[-2:] for f in features]
        all_features_anchors = self.grid_anchors(grid_sizes)
        anchors = []
        for img_h, img_w in image_list.image_sizes:
            anchors_in_one_image = []
            for one_feature_anchors in all_features_anchors:
                anchors_in_one_image.append(
                    BoxList(one_feature_anchors.to(device), (img_w, img_h), mode="xyxy")
                )
            anchors.append(anchors_in_one_image)
        return anchors


def generate_anchors(sizes=[32, 64, 128, 256, 512], ratios=[0.5, 1, 2]):
    anchors = []
    for size in sizes:
        anchors.append(torch.from_numpy(_generate_anchors(size, ratios)))

    return anchors


def _generate_anchors(size, ratios):
    anchor = np.array([0, 0, size - 1, size - 1], dtype=np.float32)
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * np.sqrt(ratios)
    hs = h / np.sqrt(ratios)

    return np.vstack(
        (
            x_ctr - 0.5 * (ws - 1),
            y_ctr - 0.5 * (hs - 1),
            x_ctr + 0.5 * (ws - 1),
            y_ctr + 0.5 * (hs - 1),
        )
    ).T


def _whctrs(anchor):
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr
