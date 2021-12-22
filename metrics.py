import torch
import torch.nn as nn
from torchvision.ops.boxes import _box_inter_union

class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, pred, target):
        smooth = 20
        m1 = pred.view(-1)
        m2 = target.view(-1)
        # inter = (m1 * m2).sum()
        # union = (m1 + m2).sum() - inter
        # score = (inter + smooth) / (union + smooth)
        score, _, _ = getIoU(m1, m2)

        return 1 - score

class GIoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, pred, target):
        m1 = pred.view(-1)
        m2 = target.view(-1)

        iou, _, union = getIoU(m1, m2)

        # area of the smallest enclosing box
        min_box = torch.min(m1, m2)
        max_box = torch.max(m1, m2)
        area_c = (max_box - min_box).sum()

        giou = iou - ((area_c - union) / (area_c + 1e-7))

        loss = 1 - giou

        return loss


def getIoU(m1, m2):
    smooth = 20
    inter = (m1 * m2).sum()
    union = (m1 + m2).sum() - inter
    IoU = (inter + smooth) / (union + smooth)
    return IoU, inter, union
