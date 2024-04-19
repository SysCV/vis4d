"""Cross entropy loss."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

from .base import Loss
from .reducer import LossReducer, mean_loss


class CrossEntropyLoss(Loss):
    """Cross entropy loss class."""

    def __init__(
        self,
        reducer: LossReducer = mean_loss,
        class_weights: list[float] | None = None,
    ) -> None:
        """Creates an instance of the class.

        Args:
            reducer (LossReducer): Reducer for the loss function. Defaults to
                mean_loss.
            class_weights (list[float], optional): Class weights for the loss
                function. Defaults to None.
        """
        super().__init__(reducer)
        self.class_weights = class_weights

    def forward(
        self,
        output: Tensor,
        target: Tensor,
        reducer: LossReducer | None = None,
        ignore_index: int = 255,
    ) -> Tensor:
        """Forward pass.

        Args:
            output (list[Tensor]): Model output.
            target (Tensor): Assigned segmentation target mask.
            reducer (LossReducer, optional): Reducer for the loss function.
                Defaults to None.
            ignore_index (int): Ignore class id. Default to 255.

        Returns:
            Tensor: Computed loss.
        """
        if self.class_weights is not None:
            class_weights = output.new_tensor(
                self.class_weights, device=output.device
            )
        else:
            class_weights = None
        reducer = reducer or self.reducer

        return reducer(
            cross_entropy(
                output, target, class_weights, ignore_index=ignore_index
            )
        )


def cross_entropy(
    output: Tensor,
    target: Tensor,
    class_weights: Tensor | None = None,
    ignore_index: int = 255,
) -> Tensor:
    """Cross entropy loss function.

    Args:
        output (Tensor): Model output.
        target (Tensor): Assigned segmentation target mask.
        class_weights (Tensor | None, optional): Class weights for the loss
            function. Defaults to None.
        ignore_index (int): Ignore class id. Default to 255.

    Returns:
        Tensor: Computed loss.
    """
    return F.cross_entropy(
        output,
        target.long(),
        weight=class_weights,
        ignore_index=ignore_index,
        reduction="none",
    )
