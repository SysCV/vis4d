"""Multi-positive cross entropy loss."""
import torch

from .base import Loss
from .reducer import LossReducer, SumWeightedLoss


class MultiPosCrossEntropyLoss(Loss):
    """Multi-positive cross entropy loss class."""

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
        avg_factor: float,
    ) -> torch.Tensor:
        """Multi-positive cross entropy loss forward."""
        return multi_pos_cross_entropy(
            pred,
            target,
            reducer=SumWeightedLoss(weight, avg_factor),
        )


def multi_pos_cross_entropy(
    pred: torch.Tensor,
    target: torch.Tensor,
    reducer: LossReducer,
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

    return reducer(loss)