import torch.nn as nn

class FPNPredictor(nn.Module):
    def __init__(self, in_channels, num_classes, num_bbox_reg_classes):
        super(FPNPredictor, self).__init__()
        
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_bbox_reg_classes*4)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.cls_score.bias, 0)
        nn.init.constant_(self.bbox_pred.bias, 0)
    
    def forward(self, x):
        scores = self.cls_score(x)
        bbox_reg = self.bbox_pred(x)

        return scores, bbox_reg

def make_roi_box_predictor(cfg, in_channels):
    num_classes=cfg["MODEL"]["ROI_BOX_HEAD"]["NUM_CLASSES"]
    num_bbox_reg_classes= 2 if cfg["MODEL"]["CLS_AGNOSTIC_BBOX_REG"] else num_classes

    return FPNPredictor(
        in_channels,
        num_classes=num_classes,
        num_bbox_reg_classes=num_bbox_reg_classes
    )

        