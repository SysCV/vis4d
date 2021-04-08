"""Multi-positive cross entropy loss."""
from typing import Optional

import torch

from .base import BaseLoss, LossConfig
from .utils import weight_reduce_loss


class MultiPosCrossEntropyLoss(BaseLoss):
    """Multi-positive cross entropy loss class."""

    def __init__(self, cfg: LossConfig):
        """Init."""
        super().__init__()
        self.cfg = cfg

    def forward(  # type: ignore # pylint: disable=arguments-differ
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        reduction_override: Optional[str] = None,
        avg_factor: Optional[float] = None,
    ) -> torch.Tensor:
        """Multi-positive cross entropy loss forward."""
        assert pred.size() == target.size()
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = (
            reduction_override
            if reduction_override is not None
            else self.cfg.reduction
        )
        loss_cls = self.cfg.loss_weight * multi_pos_cross_entropy(
            pred,
            target,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
        )
        return loss_cls


def multi_pos_cross_entropy(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    avg_factor: Optional[float] = None,
) -> torch.Tensor:
    """Calculate multi-positive cross-entropy loss."""
    # element-wise losses
    pos_inds = (target == 1).float()
    neg_inds = (target == 0).float()
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
