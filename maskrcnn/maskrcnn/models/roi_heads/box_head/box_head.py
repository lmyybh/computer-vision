import torch
import torch.nn as nn

from .loss import make_roi_box_loss_evaluator
from .feature_extractor import make_roi_box_feature_extractor
from .predictor import make_roi_box_predictor
from .inference import make_roi_box_post_processor


class ROIBoxHead(nn.Module):
    def __init__(self, cfg, in_channels):
        super(ROIBoxHead, self).__init__()

        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_box_predictor(
            cfg, self.feature_extractor.out_channels
        )
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
           features (list[Tensor]): 长度为特征图个数，每个 tensor 为 一个 batch 图片所有该尺寸特征图
           proposals (list[BoxList]): 长度为 batch 中图片个数，每个 boxlist 代表对应图片中的所有备选框
           targets (list[Target], optional): 长度为 batch 中图片个数，每个 target 代表对应图片中的真实数据
        """
        if self.training:
            with torch.no_grad():
                proposal_targets = self.loss_evaluator.subsample(proposals, targets)
                proposals = [
                    target.get_field("proposals") for target in proposal_targets
                ]
        x = self.feature_extractor(features, proposals)
        cls_logits, bbox_reg = self.predictor(x)

        if not self.training:
            targets = self.post_processor(cls_logits, bbox_reg, proposals)
            return x, targets, {}

        loss_classifier, loss_bbox_reg = self.loss_evaluator([cls_logits], [bbox_reg])

        return (
            x,
            proposal_targets,
            dict(loss_classifier=loss_classifier, loss_bbox_reg=loss_bbox_reg),
        )


def build_roi_box_head(cfg, in_channels):
    return ROIBoxHead(cfg, in_channels)
