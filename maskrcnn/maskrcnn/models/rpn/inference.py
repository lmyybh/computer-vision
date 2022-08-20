import torch
import torch.nn as nn

from maskrcnn.models.box_coder import BoxCoder
from maskrcnn.data.target import BoxList, cat_boxlists


class RPNPostProcessor(nn.Module):
    def __init__(
        self,
        pre_top_k,
        post_nms_top_n,
        nms_threshold,
        min_size,
        fpn_post_nms_top_n=None,
        fpn_post_nms_per_batch=None,
    ):
        super(RPNPostProcessor, self).__init__()
        self.pre_top_k = pre_top_k
        self.post_nms_top_n = post_nms_top_n
        self.nms_threshold = nms_threshold
        self.min_size = min_size

        self.box_coder = BoxCoder()

        if fpn_post_nms_top_n is None:
            fpn_post_nms_top_n = post_nms_top_n
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.fpn_post_nms_per_batch = fpn_post_nms_per_batch

    def forward_for_single_feature(self, anchors, logits, bbox_reg):
        """
        Arguments:
            anchors: list[BoxList]
            logits: tensor of size [N, A, H, W]
            bbox_reg: tensor of size [N, A * 4, H, W]
        """
        device = logits.device
        N, A, H, W = logits.shape
        num_anchors = A * H * W

        # 将 logits 按像素顺序展开为 [N, -1] 大小
        logits = logits.permute(0, 2, 3, 1).reshape(N, -1)
        logits = logits.sigmoid()

        # 将 bbox_reg 按像素顺序展开为 [N, -1, 4] 大小
        bbox_reg = (
            bbox_reg.reshape(N, A, 4, H, W).permute(0, 3, 4, 1, 2).reshape(N, -1, 4)
        )

        # 将 anchors 按像素顺序展开为 [N, -1, 4] 大小
        image_sizes = [box_list.size for box_list in anchors]  # 记录 batch 中每张图片大小，用于后续使用
        anchors = torch.cat([box_list.bbox for box_list in anchors], dim=0).reshape(
            N, -1, 4
        )

        # 后处理第一部分：只保留物品概率最高的前 pre_nms_top_n 个 anchors
        pre_top_k = min(num_anchors, self.pre_top_k)
        logits, top_k_idx = logits.topk(pre_top_k, dim=1, sorted=True)

        batch_idx = torch.arange(N, device=device).reshape(-1, 1)
        bbox_reg = bbox_reg[batch_idx, top_k_idx]

        anchors = anchors[batch_idx, top_k_idx]

        proposals = self.box_coder.decode(anchors.view(-1, 4), bbox_reg.view(-1, 4))
        proposals = proposals.view(N, -1, 4)
        result = []
        for proposal, score, image_size in zip(proposals, logits, image_sizes):
            boxlist = BoxList(proposal, image_size, score, mode="xyxy")
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = boxlist.remove_small_boxes(min_size=self.min_size)
            boxlist = boxlist.boxlist_nms(self.nms_threshold, self.post_nms_top_n)
            result.append(boxlist)

        return result

    def add_gt_proposals(self, proposals, targets):
        device = proposals[0].bbox.device
        gt_boxes = []
        for target in targets:
            gt_box = target.get_field("bboxes")
            gt_scores = torch.ones(len(gt_box), device=device)
            gt_boxes.append(
                BoxList(
                    gt_box.bbox,
                    gt_box.size,
                    gt_scores,
                    gt_box.mode,
                )
            )

        return [
            cat_boxlists([proposal, gt_box])
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]

    def forward(self, anchors, logits, bbox_reg, targets=None):
        # 从 图片 -> 特征图 -> anchors 的列表嵌套方式 变为 特征图 -> 图片 -> anchors 的嵌套方式
        anchors = list(zip(*anchors))

        sampled_boxes = []
        for feature_anchors, feature_logits, feature_bbox_reg in zip(
            anchors, logits, bbox_reg
        ):
            sampled_boxes.append(
                self.forward_for_single_feature(
                    feature_anchors, feature_logits, feature_bbox_reg
                )
            )

        # 变回 图片 -> 特征图 -> anchors 的列表嵌套方式
        boxlists = list(zip(*sampled_boxes))

        # 把每张图片的所有特征图上的 boxes 合并为一个 boxlist
        boxlists = [cat_boxlists(boxes) for boxes in boxlists]

        # 如果 fpn 的特征图层级大于 1 时，再选择一次
        if len(logits) > 1:
            boxlists = self.select_over_batch(boxlists)

        if self.training and targets is not None:
            boxlists = self.add_gt_proposals(boxlists, targets)

        return boxlists

    def select_over_batch(self, boxlists):
        # 训练时，在整个 batch 上进行分数排序，选择 topk
        # 理论上应该在每个图片上单独选择 topk，但是这里是为了和 Detectron 保持一致
        # if self.training and self.fpn_post_nms_per_batch:
        #     boxlist_sizes = [boxlist.bbox.shape[0] for boxlist in boxlists]
        #     scores = torch.cat([boxlist.scores for boxlist in boxlists], dim=0)
        #     post_nms_top_n = min(len(scores), self.fpn_post_nms_top_n)
        #     _, top_k_idx = torch.topk(scores, post_nms_top_n, dim=0)

        #     masks = torch.zeros_like(scores, dtype=torch.bool)
        #     masks[top_k_idx] = 1
        #     masks = masks.split(boxlist_sizes)

        #     for i in range(len(boxlists)):
        #         boxlists[i] = boxlists[i][masks[i]]
        # else:  # 测试时，在单个图片上进行分数排序，选择 topk
        #     for i in range(len(boxlists)):
        #         post_nms_top_n = min(len(boxlists[i].scores), self.fpn_post_nms_top_n)
        #         _, top_k_idx = torch.topk(boxlists[i].scores, post_nms_top_n)
        #         boxlists[i] = boxlists[i][top_k_idx]

        # 统一在每个图片上单独选择 topk
        for i in range(len(boxlists)):
            post_nms_top_n = min(len(boxlists[i].scores), self.fpn_post_nms_top_n)
            _, top_k_idx = torch.topk(boxlists[i].scores, post_nms_top_n)
            boxlists[i] = boxlists[i][top_k_idx]

        return boxlists


def make_rpn_post_processor(cfg, is_train):
    if is_train:
        pre_top_k = cfg["MODEL"]["RPN"]["PRE_TOP_K_TRAIN"]
        post_nms_top_n = cfg["MODEL"]["RPN"]["POST_NMS_TOP_N_TRAIN"]
        fpn_post_nms_top_n = cfg["MODEL"]["RPN"]["FPN_POST_NMS_TOP_N_TRAIN"]
    else:
        pre_top_k = cfg["MODEL"]["RPN"]["PRE_TOP_K_TEST"]
        post_nms_top_n = cfg["MODEL"]["RPN"]["POST_NMS_TOP_N_TEST"]
        fpn_post_nms_top_n = cfg["MODEL"]["RPN"]["FPN_POST_NMS_TOP_N_TEST"]

    return RPNPostProcessor(
        pre_top_k=pre_top_k,
        post_nms_top_n=post_nms_top_n,
        nms_threshold=cfg["MODEL"]["RPN"]["NMS_THRESHOLD"],
        min_size=cfg["MODEL"]["RPN"]["MIN_SIZE"],
        fpn_post_nms_top_n=fpn_post_nms_top_n,
        fpn_post_nms_per_batch=cfg["MODEL"]["RPN"]["FPN_POST_NMS_PER_BATCH"],
    )
