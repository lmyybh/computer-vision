import torch


def cat(tensors, dim=0):
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def concat_box_prediction_layers(logits, bbox_reg):
    box_cls_flattened = []
    box_regression_flattened = []
    for box_cls_per_level, bbox_reg_per_level in zip(logits, bbox_reg):
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = bbox_reg_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A

        box_cls_per_level = box_cls_per_level.view(N, -1, C, H, W)
        box_cls_per_level = box_cls_per_level.permute(0, 3, 4, 1, 2)
        box_cls_per_level = box_cls_per_level.reshape(N, -1, C)
        box_cls_flattened.append(box_cls_per_level)

        bbox_reg_per_level = bbox_reg_per_level.view(N, -1, 4, H, W)
        bbox_reg_per_level = bbox_reg_per_level.permute(0, 3, 4, 1, 2)
        bbox_reg_per_level = bbox_reg_per_level.reshape(N, -1, 4)
        box_regression_flattened.append(bbox_reg_per_level)

    box_cls = cat(box_cls_flattened, dim=1).reshape(-1, C)
    box_regression = cat(box_regression_flattened, dim=1).reshape(-1, 4)

    return box_cls, box_regression
