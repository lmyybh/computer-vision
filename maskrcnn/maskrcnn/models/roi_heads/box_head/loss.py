import torch
import torch.nn as nn
import torch.nn.functional as F

from maskrcnn.data.target import boxlist_iou, Target
from maskrcnn.models.matcher import Matcher
from maskrcnn.models.box_coder import BoxCoder
from maskrcnn.models.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler,
)
from maskrcnn.models.rpn.utils import cat
from maskrcnn.models.rpn.loss import smooth_l1_loss


class BoxHeadLoss(nn.Module):
    def __init__(self, matcher, sampler, box_coder, cls_agnostic_bbox_reg=False):
        super().__init__()
        self.matcher = matcher
        self.sampler = sampler
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target.get_field("bboxes"), proposal)
        matched_idxs = self.matcher(match_quality_matrix)
        matched_targets = target[matched_idxs.clamp(min=0)]  # 这里把 IOU 低的框都匹配到了第一个真实框
        return matched_targets, matched_idxs

    def prepare_targets(self, proposals, targets):
        labels = []
        regression_targets = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets, matched_idxs = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            labels_per_image = matched_targets.get_field("labels").to(dtype=torch.int64)

            bg_idxs = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_idxs] = 0

            ignore_idxs = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_idxs] = -1

            regression_targets_per_image = self.box_coder.encode(
                matched_targets.get_field("bboxes").bbox, proposals_per_image.bbox
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

        return labels, regression_targets

    def subsample(self, proposals, targets):
        # 配对，为 proposals 中的每个框配一个真实框的 label 和 box_reg
        labels, regression_targets = self.prepare_targets(proposals, targets)
        pos_idxs, neg_idxs = self.sampler(labels)

        proposals = list(proposals)

        proposal_targets = []
        for labels_per_image, regression_targets_per_image, proposals_per_image in zip(
            labels, regression_targets, proposals
        ):
            proposal_targets_per_image = Target()
            proposal_targets_per_image.add_field("proposals", proposals_per_image)
            proposal_targets_per_image.add_field("labels", labels_per_image)
            proposal_targets_per_image.add_field(
                "regression_targets", regression_targets_per_image
            )
            proposal_targets.append(proposal_targets_per_image)

        for idx, (pos_idxs_img, neg_idxs_img) in enumerate(zip(pos_idxs, neg_idxs)):
            img_idxs = torch.nonzero(pos_idxs_img | neg_idxs_img).flatten()
            proposal_targets_per_image = proposal_targets[idx][img_idxs]
            proposal_targets[idx] = proposal_targets_per_image

        self._proposal_targets = proposal_targets
        return proposal_targets

    def forward(self, logits, bbox_reg):
        logits = cat(logits, dim=0)
        bbox_reg = cat(bbox_reg, dim=0)
        device = logits.device
        assert hasattr(self, "_proposal_targets"), "必须存在 _proposal_targets"

        proposal_targets = self._proposal_targets
        labels = cat([target.get_field("labels") for target in proposal_targets])
        regression_targets = cat(
            [target.get_field("regression_targets") for target in proposal_targets]
        )

        classification_loss = F.cross_entropy(logits, labels)

        pos_idxs_subset = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[pos_idxs_subset]
        if self.cls_agnostic_bbox_reg:
            map_idxs = torch.tensor([4, 5, 6, 7], device=device)
        else:
            map_idxs = 4 * labels_pos[:, None] + torch.tensor(
                [0, 1, 2, 3], device=device
            )

        box_loss = smooth_l1_loss(
            bbox_reg[pos_idxs_subset[:, None], map_idxs],
            regression_targets[pos_idxs_subset],
            size_average=False,
            beta=1,
        )

        box_loss = box_loss / labels.numel()

        return classification_loss, box_loss


def make_roi_box_loss_evaluator(cfg):
    matcher = Matcher(
        cfg["MODEL"]["ROI_HEADS"]["BG_IOU_THRESHOLD"],
        cfg["MODEL"]["ROI_HEADS"]["FG_IOU_THRESHOLD"],
        allow_low_quality_matches=False,
    )

    sampler = BalancedPositiveNegativeSampler(
        cfg["MODEL"]["ROI_HEADS"]["BATCH_SIZE_PER_IMAGE"],
        cfg["MODEL"]["ROI_HEADS"]["POSITIVE_FRACTION"],
    )

    box_coder = BoxCoder(weights=cfg["MODEL"]["ROI_HEADS"]["BBOX_REG_WEIGHTS"])

    return BoxHeadLoss(
        matcher,
        sampler,
        box_coder,
        cls_agnostic_bbox_reg=cfg["MODEL"]["CLS_AGNOSTIC_BBOX_REG"],
    )
