"""Segmentation cross entropy loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vis4d.common.typing import LossesType

from .base import Loss
from .reducer import LossReducer, mean_loss


class SegCrossEntropyLoss(Loss):
    """Segmentation cross entropy loss class.

    Wrapper for nn.CrossEntropyLoss that additionally clips the output to the
    target size and converts the target mask tensor to long.
    """

    def __init__(self, reducer: LossReducer = mean_loss) -> None:
        """Creates an instance of the class.

        Args:
            reducer (LossReducer): Reducer for the loss function. Defaults to
                mean_loss.
        """
        super().__init__(reducer)

    def forward(
        self, output: torch.Tensor, target: torch.Tensor
    ) -> LossesType:
        """Forward pass.

        Args:
            output (list[torch.Tensor]): Model output.
            target (torch.Tensor): Assigned segmentation target mask.

        Returns:
            LossesType: Computed loss.
        """
        losses: LossesType = {}
        losses["loss_seg"] = self.reducer(seg_cross_entropy(output, target))
        return losses


def seg_cross_entropy(
    output: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Segmentation cross entropy loss function.

    Args:
        output (torch.Tensor): Model output.
        target (torch.Tensor): Assigned segmentation target mask.

    Returns:
        torch.Tensor: Computed loss.
    """
    tgt_h, tgt_w = target.shape[-2:]
    return F.cross_entropy(
        output[:, :, :tgt_h, :tgt_w],
        target.long(),
        ignore_index=255,
        reduction="none",
    )
