import torch
import torch.nn as nn
import torch.nn.functional as F

from maskrcnn.models.box_coder import BoxCoder
from maskrcnn.data.target import Target, BoxList, cat_boxlists


class PostProcessor(nn.Module):
    def __init__(
        self,
        score_threshold=0.05,
        nms=0.5,
        detections_per_img=100,
        box_coder=None,
        cls_agnostic_bbox_reg=False,
    ):
        super().__init__()
        self.score_threshold = score_threshold
        self.nms = nms
        self.detections_per_img = detections_per_img
        if box_coder is None:
            box_coder = BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg

    def forward(self, cls_logits, bbox_reg, boxes):
        cls_prob = F.softmax(cls_logits, dim=-1)

        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        concat_boxes = torch.cat([b.bbox for b in boxes], dim=0)

        if self.cls_agnostic_bbox_reg:
            bbox_reg = bbox_reg[:, -4:]

        proposals = self.box_coder.decode(
            bbox_reg.view(sum(boxes_per_image), -1), concat_boxes
        )

        if self.cls_agnostic_bbox_reg:
            proposals = proposals.repeat(1, cls_prob.shape[1])

        num_classes = cls_prob.shape[1]

        proposals = proposals.split(boxes_per_image, dim=0)
        cls_prob = cls_prob.split(boxes_per_image, dim=0)

        result = []
        for prob, proposals_per_img, image_shape in zip(
            cls_prob, proposals, image_shapes
        ):
            boxlist = self.prepare_boxlist(proposals_per_img, prob, image_shape)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            target = self.filter_results(boxlist, num_classes)
            result.append(target)
        return result

    def prepare_boxlist(self, boxes, scores, image_shape):
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        return BoxList(boxes, image_shape, scores, mode="xyxy")

    def filter_results(self, boxlist, num_classes):
        boxes = boxlist.bbox.reshape(-1, num_classes * 4)
        scores = boxlist.scores.reshape(-1, num_classes)

        device = scores.device
        boxlists, all_labels = [], []
        all_idxs = scores > self.score_threshold
        for j in range(1, num_classes):
            idxs = all_idxs[:, j].nonzero().squeeze(1)
            scores_j = scores[idxs, j]
            boxes_j = boxes[idxs, j * 4 : (j + 1) * 4]
            boxlist_for_class = BoxList(boxes_j, boxlist.size, scores_j, mode="xyxy")
            boxlist_for_class = boxlist_for_class.boxlist_nms(self.nms)
            num_labels = len(boxlist_for_class)
            labels = torch.full((num_labels,), j, dtype=torch.int64, device=device)
            boxlists.append(boxlist_for_class)
            all_labels.append(labels)

        num_detections = len(boxlists)
        boxlist = cat_boxlists(boxlists)
        labels = torch.cat(all_labels)
        result = Target()
        result.add_field("bboxes", boxlist)
        result.add_field("labels", labels)

        if num_detections > self.detections_per_img > 0:
            scores = boxlist.scores
            img_threshold, _ = torch.kthvalue(
                scores.cpu(), num_detections - self.detections_per_img + 1
            )
            keep = scores >= img_threshold.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]
        return result


def make_roi_box_post_processor(cfg):
    box_coder = BoxCoder(weights=cfg["MODEL"]["ROI_HEADS"]["BBOX_REG_WEIGHTS"])
    return PostProcessor(
        score_threshold=cfg["MODEL"]["ROI_HEADS"]["SCORE_THRESHOLD"],
        nms=cfg["MODEL"]["ROI_HEADS"]["NMS"],
        detections_per_img=cfg["MODEL"]["ROI_HEADS"]["DETECTIONS_PER_IMG"],
        box_coder=box_coder,
        cls_agnostic_bbox_reg=cfg["MODEL"]["CLS_AGNOSTIC_BBOX_REG"],
    )
