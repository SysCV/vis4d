"""Multi-positive cross entropy loss."""
from typing import Optional

import torch
import torch.nn.functional as F

from .base import BaseLoss, LossConfig
from .utils import weight_reduce_loss


class CrossEntropyLoss(BaseLoss):
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
        """Cross entropy loss forward."""
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = (
            reduction_override
            if reduction_override is not None
            else self.cfg.reduction
        )
        loss_cls = self.cfg.loss_weight * cross_entropy(
            pred,
            target,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
        )
        return loss_cls


def cross_entropy(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    avg_factor: Optional[float] = None,
) -> torch.Tensor:
    """Calculate cross-entropy loss."""
    loss = F.cross_entropy(
        pred,
        target,
        reduction="none",
    )

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()  # pragma: no cover
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor
    )

    return loss
