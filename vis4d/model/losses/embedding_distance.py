"""Embedding distance loss."""
from typing import Optional, Tuple

import torch

from vis4d.common.bbox.utils import random_choice

from .base import BaseLoss
from .utils import l2_loss


class EmbeddingDistanceLoss(BaseLoss):
    """Embedding distance loss."""

    def __init__(
        self,
        reduction: str = "mean",
        loss_weight: Optional[float] = 1.0,
        neg_pos_ub: float = 3.0,
        pos_margin: float = 0.0,
        neg_margin: float = 0.3,
        hard_mining: bool = True,
    ):
        """Init."""
        super().__init__(reduction, loss_weight)
        self.neg_pos_ub = neg_pos_ub
        self.neg_margin = neg_margin
        self.pos_margin = pos_margin
        self.hard_mining = hard_mining

    def __call__(  # type: ignore # pylint: disable=arguments-differ
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        reduction_override: Optional[str] = None,
    ) -> torch.Tensor:
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            loss_bbox (torch.Tensor): embedding distance loss.
        """
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = (
            reduction_override
            if reduction_override is not None
            else self.reduction
        )
        pred, weight, avg_factor = self.update_weight(pred, target, weight)
        loss = self.loss_weight * l2_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor
        )
        return loss

    def update_weight(
        self, pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Update element-wise loss weights.

        Exclude negatives according to maximum fraction of samples and/or
        hard negative mining.
        """
        if weight is None:
            weight = target.new_ones(target.size())
        invalid_inds = weight <= 0
        target[invalid_inds] = -1
        pos_inds = target == 1
        neg_inds = target == 0

        if self.pos_margin > 0:
            pred[pos_inds] -= self.pos_margin
        if self.neg_margin > 0:
            pred[neg_inds] -= self.neg_margin
        pred = torch.clamp(pred, min=0, max=1)

        num_pos = max(1, int((target == 1).sum()))
        num_neg = int((target == 0).sum())
        if self.neg_pos_ub > 0 and num_neg / num_pos > self.neg_pos_ub:
            num_neg = int(num_pos * self.neg_pos_ub)
            neg_idx = torch.nonzero(target == 0, as_tuple=False)

            if self.hard_mining:
                costs = l2_loss(pred, target, reduction="none")[
                    neg_idx[:, 0], neg_idx[:, 1]
                ].detach()
                neg_idx = neg_idx[costs.topk(num_neg)[1], :]
            else:
                neg_idx = random_choice(neg_idx, num_neg)

            new_neg_inds = neg_inds.new_zeros(neg_inds.size()).bool()
            new_neg_inds[neg_idx[:, 0], neg_idx[:, 1]] = True

            invalid_neg_inds = torch.logical_xor(neg_inds, new_neg_inds)
            weight[invalid_neg_inds] = 0

        avg_factor = (weight > 0).sum()
        return pred, weight, avg_factor
