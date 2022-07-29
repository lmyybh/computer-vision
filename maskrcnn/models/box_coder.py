import math
import torch


class BoxCoder:
    def __init__(
        self, weights=(1.0, 1.0, 1.0, 1.0), bbox_xform_clip=math.log(1000.0 / 16)
    ):
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def decode(self, anchors, bbox_reg):
        anchors = anchors.to(bbox_reg.dtype)

        # 这里使用 0::4 可以使得 dx 等 tensor 的形状为 [4, 1]
        # 只使用 0 做索引，形状为 (4,)
        widths = anchors[:, 2::4] - anchors[:, 0::4] + 1
        heights = anchors[:, 3::4] - anchors[:, 1::4] + 1
        ctr_x = anchors[:, 0::4] + 0.5 * widths
        ctr_y = anchors[:, 1::4] + 0.5 * heights

        wx, wy, ww, wh = self.weights

        dx = bbox_reg[:, 0::4] / wx
        dy = bbox_reg[:, 1::4] / wy
        dw = bbox_reg[:, 2::4] / ww
        dh = bbox_reg[:, 3::4] / wh

        # clamp
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        # pred_boxes
        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * heights + ctr_y
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        pred_boxes = torch.zeros_like(anchors)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1

        return pred_boxes
