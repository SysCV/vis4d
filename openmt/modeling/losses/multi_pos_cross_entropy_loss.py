import torch

from .base_loss import BaseLoss, LossConfig
from .utils import weight_reduce_loss


class MultiPosCrossEntropyLoss(BaseLoss):
    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        cls_score,
        label,
        weight=None,
        avg_factor=None,
        reduction_override=None,
        **kwargs
    ):
        assert cls_score.size() == label.size()
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = (
            reduction_override if reduction_override else self.cfg.reduction
        )
        loss_cls = self.cfg.loss_weight * multi_pos_cross_entropy(
            cls_score,
            label,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
        )
        return loss_cls


def multi_pos_cross_entropy(
    pred, label, weight=None, reduction="mean", avg_factor=None
):
    # element-wise losses
    pos_inds = (label == 1).float()
    neg_inds = (label == 0).float()
    exp_pos = (torch.exp(-1 * pred) * pos_inds).sum(dim=1)
    exp_neg = (torch.exp(pred.clamp(max=80)) * neg_inds).sum(dim=1)
    loss = torch.log(1 + exp_pos * exp_neg)

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor
    )

    return loss
