__author__ = 'Hao Bian'

import torch
import torch.nn as nn
import torchmetrics
from torchmetrics import metric

# from .boundary_loss import BDLoss, SoftDiceLoss, DC_and_BD_loss, HDDTBinaryLoss,\
#      DC_and_HDBinary_loss, DistBinaryDiceLoss
# from .dice_loss import GDiceLoss, GDiceLossV2, SSLoss, SoftDiceLoss,\
#      IoULoss, TverskyLoss, FocalTversky_loss, AsymLoss, DC_and_CE_loss,\
#          PenaltyGDiceLoss, DC_and_topk_loss, ExpLog_loss
# from .focal_loss import FocalLoss
# from .hausdorff import HausdorffDTLoss, HausdorffERLoss
# from .lovasz_loss import LovaszSoftmax
# from .ND_Crossentropy import CrossentropyND, TopKLoss, WeightedCrossEntropyLoss,\
#      WeightedCrossEntropyLossV2, DisPenalizedCE

from pytorch_toolbelt import losses as L


def create_loss(args, w1=1.0, w2=0.5):
    conf_loss = args.name
    # MulticlassJaccardLoss(classes=np.arange(11)
    # mode = args.base_loss #BINARY_MODE \MULTICLASS_MODE \MULTILABEL_MODE
    loss = None
    # 以下是多类的loss
    if conf_loss.split('_')[0] == 'multiclass':
        if conf_loss == 'multiclass_CE_Dice':
            loss = L.JointLoss(nn.CrossEntropyLoss(),
                               L.DiceLoss(mode='multiclass'), w1, w2)
        else:
            assert False and "Invalid multi class loss"
    else:
        if hasattr(nn, conf_loss):  # 判断对象是否包含对应的属性
            loss = getattr(nn, conf_loss)()  # 获取对象的属性值
        # 以下均是二值的loss
        elif conf_loss == "BCE":
            w = torch.Tensor([1 - 0.35, 0.35])
            loss = nn.CrossEntropyLoss(w)
        elif conf_loss == "focal":
            loss = L.BinaryFocalLoss()
        elif conf_loss == "jaccard":
            loss = L.BinaryJaccardLoss()
        elif conf_loss == "jaccard_log":
            loss = L.BinaryJaccardLoss()
        elif conf_loss == "dice":
            loss = L.BinaryDiceLoss()
        elif conf_loss == "dice_log":
            loss = L.BinaryDiceLogLoss()
        elif conf_loss == "dice_log":
            loss = L.BinaryDiceLogLoss()
        elif conf_loss == "bce+lovasz":
            loss = L.JointLoss(BCEWithLogitsLoss(),
                               L.BinaryLovaszLoss(), w1, w2)
        elif conf_loss == "lovasz":
            loss = L.BinaryLovaszLoss()
        elif conf_loss == "bce+jaccard":
            loss = L.JointLoss(BCEWithLogitsLoss(),
                               L.BinaryJaccardLoss(), w1, w2)
        elif conf_loss == "bce+log_jaccard":
            loss = L.JointLoss(BCEWithLogitsLoss(),
                               L.BinaryJaccardLogLoss(), w1, w2)
        elif conf_loss == "bce+log_dice":
            loss = L.JointLoss(BCEWithLogitsLoss(),
                               L.BinaryDiceLogLoss(), w1, w2)
        elif conf_loss == "reduced_focal":
            loss = L.BinaryFocalLoss(reduced=True)
        else:
            assert False and "Invalid binary loss"
            raise ValueError
    return loss


def create_metric(metric_name, w1=1.0, w2=0.5):
    conf_metric = metric_name

    # 以下是多类的loss
    if conf_metric.split('_')[0] == 'multiclass':
        if conf_metric == 'multiclass_jaccard':
            metric = L.JaccardLoss(mode='multiclass')
        elif conf_metric == 'multiclass_dice':
            metric = L.DiceLoss(mode='multiclass', from_logits=False)
        else:
            assert False and "Invalid multi class loss"
    else:
        if conf_metric == "AUC":
            metric = torchmetrics.AUROC(pos_label=1)
        elif conf_metric == "ACC":
            metric = torchmetrics.Accuracy()
        else:
            assert False and "Invalid binary class loss"

    return metric


if __name__ == '__main__':
    logits = torch.load('/data/bianhao/code/GNN/seg-gini-main/logits.pt')
    targets = torch.load('/data/bianhao/code/GNN/seg-gini-main/targets.pt')
    dice = create_metric('multiclass_dice')
    print(dice(logits[0], targets[0]))
