"""Segmentation cross entropy loss."""

from __future__ import annotations

from torch import Tensor

from vis4d.common.typing import LossesType

from .base import Loss
from .cross_entropy import cross_entropy
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
        self, output: Tensor, target: Tensor, ignore_index: int = 255
    ) -> LossesType:
        """Forward pass.

        Args:
            output (list[Tensor]): Model output.
            target (Tensor): Assigned segmentation target mask.
            ignore_index (int): Ignore class id. Default to 255.

        Returns:
            LossesType: Computed loss.
        """
        losses: LossesType = {}
        tgt_h, tgt_w = target.shape[-2:]
        losses["loss_seg"] = self.reducer(
            cross_entropy(
                output[:, :, :tgt_h, :tgt_w], target, ignore_index=ignore_index
            )
        )
        return losses
