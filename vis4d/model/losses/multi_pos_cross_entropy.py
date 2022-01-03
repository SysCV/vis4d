"""Multi-positive cross entropy loss."""
from typing import Optional

import torch

from .base import BaseLoss
from .utils import weight_reduce_loss


class MultiPosCrossEntropyLoss(BaseLoss):
    """Multi-positive cross entropy loss class."""

    def __call__(  # type: ignore # pylint: disable=arguments-differ
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
            else self.reduction
        )
        loss_cls = self.loss_weight * multi_pos_cross_entropy(
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
    pos_inds = target == 1
    neg_inds = target == 0
    pred_pos = pred * pos_inds.float()
    pred_neg = pred * neg_inds.float()
    # use -inf to mask out unwanted elements.
    pred_pos[neg_inds] = pred_pos[neg_inds] + float("inf")
    pred_neg[pos_inds] = pred_neg[pos_inds] + float("-inf")

    _pos_expand = torch.repeat_interleave(pred_pos, pred.shape[1], dim=1)
    _neg_expand = pred_neg.repeat(1, pred.shape[1])

    x = torch.nn.functional.pad(
        (_neg_expand - _pos_expand), (0, 1), "constant", 0
    )
    loss = torch.logsumexp(x, dim=1)

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor
    )

    return loss
