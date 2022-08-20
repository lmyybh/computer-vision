import math
import torch


class BoxCoder:
    def __init__(
        self, weights=(1.0, 1.0, 1.0, 1.0), bbox_xform_clip=math.log(1000.0 / 16)
    ):
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, ref_boxes, proposals):
        ex_widths = proposals[:, 2] - proposals[:, 0] + 1
        ex_heights = proposals[:, 3] - proposals[:, 1] + 1
        ex_ctr_x = proposals[:, 0] + 0.5 * ex_widths
        ex_ctr_y = proposals[:, 1] + 0.5 * ex_heights

        gt_widths = ref_boxes[:, 2] - ref_boxes[:, 0] + 1
        gt_heights = ref_boxes[:, 3] - ref_boxes[:, 1] + 1
        gt_ctr_x = ref_boxes[:, 0] + 0.5 * gt_widths
        gt_ctr_y = ref_boxes[:, 1] + 0.5 * gt_heights

        wx, wy, ww, wh = self.weights

        target_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        target_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
        target_dw = ww * torch.log(gt_widths / ex_widths)
        target_dh = wh * torch.log(gt_heights / ex_heights)

        return torch.stack((target_dx, target_dy, target_dw, target_dh), dim=1)

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
