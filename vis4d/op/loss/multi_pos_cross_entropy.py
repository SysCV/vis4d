"""Multi-positive cross entropy loss."""

import torch
from torch import Tensor

from .base import Loss
from .reducer import LossReducer, SumWeightedLoss


class MultiPosCrossEntropyLoss(Loss):
    """Multi-positive cross entropy loss.

    Used for appearance similiary learning in QDTrack.
    """

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        weight: Tensor,
        avg_factor: float,
    ) -> Tensor:
        """Multi-positive cross entropy loss.

        Args:
            pred (Tensor): Similarity scores before softmax. Shape [N, M]
            target (Tensor): Target for each pair. Either one, meaning
                same identity or zero, meaning different identity. Shape [N, M]
            weight (Tensor): The weight of loss for each prediction.
            avg_factor (float): Averaging factor for the loss.

        Returns:
            Tensor: Scalar loss value.
        """
        return multi_pos_cross_entropy(
            pred, target, reducer=SumWeightedLoss(weight, avg_factor)
        )


def multi_pos_cross_entropy(
    pred: Tensor, target: Tensor, reducer: LossReducer
) -> Tensor:
    """Calculate multi-positive cross-entropy loss."""
    pos_inds = torch.eq(target, 1)
    neg_inds = torch.eq(target, 0)
    pred_pos = pred * pos_inds.float()
    pred_neg = pred * neg_inds.float()
    # use -inf to mask out unwanted elements.
    pred_pos[neg_inds] = pred_pos[neg_inds] + float("inf")
    pred_neg[pos_inds] = pred_neg[pos_inds] + float("-inf")

    _pos_expand = torch.repeat_interleave(pred_pos, pred.shape[1], dim=1)
    _neg_expand = pred_neg.repeat(1, pred.shape[1])

    x = torch.nn.functional.pad(  # pylint: disable=not-callable
        (_neg_expand - _pos_expand), (0, 1), "constant", 0
    )
    loss = torch.logsumexp(x, dim=1)

    return reducer(loss)
