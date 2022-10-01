import torch
import torch.nn as nn
import torch.nn.functional as F

from yolo.data.boxmanager import BoxManager, cat_boxmgrs, boxmgr_iou


def get_predict_bboxes(feature, anchors, scale):
    device = feature.device
    w, h, num_boxes, num_features = feature.shape
    predicts = feature.reshape(-1, num_features)
    txywh = predicts[:, :4]
    objs = torch.sigmoid(predicts[:, 4]).flatten()
    classes = torch.sigmoid(predicts[:, 5:])
    anchors = torch.tensor(anchors).repeat((w * h, 1)).to(device)
    grids_idxs = torch.arange(w * h * num_boxes).reshape(-1, 1).to(device)
    cxy = torch.cat(
        (
            torch.div(grids_idxs, h * num_boxes, rounding_mode="floor"),
            torch.div(grids_idxs, num_boxes, rounding_mode="floor") % h,
        ),
        dim=1,
    ).to(device, torch.float)
    cxy += torch.sigmoid(txywh[:, :2])
    cxy *= scale
    wh = anchors * torch.exp(txywh[:, 2:4])

    cxywh = torch.cat((cxy, wh), dim=1)
    return BoxManager(cxywh, None, labels=classes, scores=objs, mode="cxywh"), predicts


def wh_iou(wh1, wh2):
    areas1 = wh1[:, 0] * wh1[:, 1]
    areas2 = wh2[:, 0] * wh2[:, 1]
    ious = torch.zeros(wh1.shape[0], wh2.shape[0]).to(wh1.device)
    for i in range(wh1.shape[0]):
        for j in range(wh2.shape[0]):
            intersection = min(wh1[i][0], wh2[j][0]) * min(wh1[i][1], wh2[j][1])
            ious[i][j] = intersection / (areas1[i] + areas2[j] - intersection)
    return ious


def positive_losses(
    features_predicts, target_boxmgr, anchors, scales, sizes, num_boxes
):
    device = features_predicts.device
    anchors = torch.tensor(anchors).to(device)
    target_boxmgr = target_boxmgr.convert("xywh")
    target_wh = target_boxmgr.bbox[:, -2:]

    anchors_indices = wh_iou(target_wh, anchors).max(dim=1).indices
    bboxes_indices = anchors_indices % 3
    matched_anchors = anchors[anchors_indices]
    features_indices = torch.div(anchors_indices, num_boxes, rounding_mode="floor")
    features_scales = torch.tensor(scales)[features_indices].to(device)
    features_sizes = torch.tensor(sizes)[features_indices].to(device)
    num_boxes_every_features = torch.tensor(sizes).prod(dim=1) * num_boxes

    scaled_bboxes_cxy = target_boxmgr.bbox[:, :2] / features_scales.reshape(-1, 1)
    top_left_xy = torch.div(scaled_bboxes_cxy, 1, rounding_mode="floor")
    gt_txy = scaled_bboxes_cxy - top_left_xy  # 取小数部分
    gt_txy[gt_txy == 0] = 1.0
    top_left_xy = scaled_bboxes_cxy - gt_txy

    gt_twh = torch.log(target_boxmgr.bbox[:, 2:4] / matched_anchors)
    gt_txywh = torch.cat([gt_txy, gt_twh], dim=1)

    grid_idxs = torch.cat([top_left_xy, bboxes_indices.reshape(-1, 1)], dim=1)

    # 计算属于第几个预测框
    start_grids_idxs = torch.tensor(
        [
            num_boxes_every_features[:fi].sum() if fi > 0 else 0
            for fi in features_indices
        ]
    ).to(device)
    positive_predict_bboxes_indices = (
        start_grids_idxs
        + (grid_idxs[:, 0] * features_sizes[:, 1] + grid_idxs[:, 1]) * num_boxes
        + grid_idxs[:, 2]
    )
    positive_predict_bboxes_indices = positive_predict_bboxes_indices.to(torch.long)

    pos_predicts = features_predicts[positive_predict_bboxes_indices]
    pred_txywh = torch.cat(
        [torch.sigmoid(pos_predicts[:, :2]), pos_predicts[:, 2:4]], dim=1
    )

    lbox = torch.mean(
        (2 - gt_twh[:, 0][:, None] * gt_twh[:, 1][:, None])
        * (pred_txywh - gt_txywh) ** 2
    )

    # 会出现 -inf，暂未找到原因
    if torch.isinf(lbox):
        lbox = torch.zeros_like(lbox)

    lobj = F.binary_cross_entropy(
        torch.sigmoid(pos_predicts[:, 4]), target_boxmgr.scores, reduction="mean"
    )
    gt_classes = F.one_hot(
        target_boxmgr.labels.to(torch.long), num_classes=pos_predicts[:, 5:].shape[1]
    ).float()

    lcls = F.binary_cross_entropy(
        torch.sigmoid(pos_predicts[:, 5:]), gt_classes, reduction="mean"
    )

    return lbox, lobj, lcls, positive_predict_bboxes_indices


class YoloV3Loss(nn.Module):
    def __init__(
        self,
        all_anchors,
        scales,
        sizes,
        noobj_threshold,
        lambda_coord,
        lambda_class,
        lambda_obj,
        lambda_noobj,
    ):
        super(YoloV3Loss, self).__init__()
        self.all_anchors = all_anchors
        self.scales = scales
        self.sizes = sizes
        self.noobj_threshold = noobj_threshold
        self.lambda_coord = lambda_coord
        self.lambda_class = lambda_class
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj

    def forward(self, features, targets):
        batch_features = [list(f) for f in zip(*features)]
        num_boxes = batch_features[0][0].shape[2]

        num_anchors_per_feature = len(self.all_anchors) // len(features)
        features_anchors = [
            self.all_anchors[i : i + num_anchors_per_feature]
            for i in range(0, len(self.all_anchors), num_anchors_per_feature)
        ]

        losses_dict = {"lbox": 0, "lcls": 0, "lobj": 0, "lnoobj": 0}
        for multi_features, target_boxmgr in zip(batch_features, targets):
            boxmgrs = []
            features_predicts = []
            for feature, anchors, scale in zip(
                multi_features, features_anchors, self.scales
            ):
                boxmgr, predicts = get_predict_bboxes(feature, anchors, scale)
                boxmgrs.append(boxmgr)
                features_predicts.append(predicts)
            predict_boxmgr = cat_boxmgrs(boxmgrs)
            features_predicts = torch.cat(features_predicts, dim=0)

            # 正样本
            lbox, lobj, lcls, positive_predict_bboxes_indices = positive_losses(
                features_predicts,
                target_boxmgr,
                self.all_anchors,
                self.scales,
                self.sizes,
                num_boxes,
            )

            # 负样本
            pt_max_ious = boxmgr_iou(predict_boxmgr, target_boxmgr).max(dim=1).values
            noobj_indices = torch.argwhere(pt_max_ious < self.noobj_threshold).flatten()

            pos_set = set(positive_predict_bboxes_indices.cpu().numpy())
            noobj_indices = [i for i in noobj_indices if i not in pos_set]

            noobjs = predict_boxmgr[noobj_indices].scores
            lnoobj = F.binary_cross_entropy(
                noobjs, torch.zeros_like(noobjs), reduction="mean"
            )

            losses_dict["lbox"] += self.lambda_coord * lbox
            losses_dict["lcls"] += self.lambda_class * lcls
            losses_dict["lobj"] += self.lambda_obj * lobj
            losses_dict["lnoobj"] += self.lambda_noobj * lnoobj

        losses_dict = {k: v / len(batch_features) for k, v in losses_dict.items()}

        return losses_dict


def build_yolov3_loss(cfg):
    return YoloV3Loss(
        all_anchors=cfg["MODEL"]["ANCHORS"],
        scales=cfg["MODEL"]["SCALES"],
        sizes=[
            [cfg["DATA"]["IMAGE_SIZE"] // scale] * 2 for scale in cfg["MODEL"]["SCALES"]
        ],
        noobj_threshold=cfg["LOSS"]["NOOBJ_THRESHOLD"],
        lambda_coord=cfg["LOSS"]["LAMBDA_COORD"],
        lambda_class=cfg["LOSS"]["LAMBDA_CLASS"],
        lambda_obj=cfg["LOSS"]["LAMBDA_OBJ"],
        lambda_noobj=cfg["LOSS"]["LAMBDA_NOOBJ"],
    )
