import torch
import torch.nn as nn
import torch.nn.functional as F

from maskrcnn.data.target import cat_boxlists, boxlist_iou
from ..matcher import Matcher
from ..box_coder import BoxCoder
from ..balanced_positive_negative_sampler import BalancedPositiveNegativeSampler
from ..rpn.utils import concat_box_prediction_layers


def generate_rpn_labels(matched_idxs):
    return matched_idxs >= 0


class RPNLoss(nn.Module):
    def __init__(self, matcher, sampler):
        super().__init__()
        self.matcher = matcher
        self.sampler = sampler
        self.boxcoder = BoxCoder()
        self.discard_cases = ["not_visibility", "between_thresholds"]

    def match_targets_to_anchors(self, anchor, target):
        target_boxlist = target.get_field("bboxes")
        match_quality_matrix = boxlist_iou(target_boxlist, anchor)
        matched_idxs = self.matcher(match_quality_matrix)

        matched_targets_boxlist = target_boxlist[matched_idxs.clamp(min=0)]

        return matched_targets_boxlist, matched_idxs

    def prepare_targets(self, anchors, targets):
        labels = []
        regression_targets = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            matched_targets_boxlist, matched_idxs = self.match_targets_to_anchors(
                anchors_per_image, targets_per_image
            )

            labels_per_image = generate_rpn_labels(matched_idxs).to(dtype=torch.float32)

            bg_indices = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_indices] = 0

            if "between_thresholds" in self.discard_cases:
                inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1

            regression_targets_per_image = self.boxcoder.encode(
                matched_targets_boxlist.bbox, anchors_per_image.bbox
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

        return labels, regression_targets

    def forward(self, anchors, logits, bbox_reg, targets):
        anchors = [cat_boxlists(anchors_per_image) for anchors_per_image in anchors]
        labels, regression_targets = self.prepare_targets(anchors, targets)

        pos_idx, neg_idx = self.sampler(labels)
        pos_idx = torch.nonzero(torch.cat(pos_idx, dim=0)).squeeze(1)
        neg_idx = torch.nonzero(torch.cat(neg_idx, dim=0)).squeeze(1)
        sampled_idx = torch.cat([pos_idx, neg_idx], dim=0)

        logits, bbox_reg = concat_box_prediction_layers(logits, bbox_reg)
        logits = logits.squeeze()

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        logits_loss = F.binary_cross_entropy_with_logits(
            logits[sampled_idx], labels[sampled_idx]
        )

        box_loss = (
            smooth_l1_loss(
                bbox_reg[sampled_idx],
                regression_targets[sampled_idx],
                size_average=False,
            )
            / sampled_idx.numel()
        )

        return logits_loss, box_loss


def smooth_l1_loss(input, target, beta=1.0 / 9, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n**2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()


def make_rpn_loss_evaluator(cfg):
    matcher = Matcher(
        cfg["MODEL"]["RPN"]["BG_IOU_THRESHOLD"],  # 0.3
        cfg["MODEL"]["RPN"]["FG_IOU_THRESHOLD"],  # 0.7
        allow_low_quality_matches=True,
    )

    sampler = BalancedPositiveNegativeSampler(
        cfg["MODEL"]["RPN"]["BATCH_SIZE_PER_IMAGE"],
        cfg["MODEL"]["RPN"]["POSITIVE_FRACTION"],
    )

    return RPNLoss(matcher, sampler)
