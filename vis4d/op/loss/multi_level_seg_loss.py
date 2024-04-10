"""Multi-level segmentation loss."""

from __future__ import annotations

from collections.abc import Callable

import torch

from vis4d.common.typing import LossesType

from .base import Loss
from .reducer import LossReducer, mean_loss
from .seg_cross_entropy_loss import seg_cross_entropy


class MultiLevelSegLoss(Loss):
    """Multi-level segmentation loss class.

    Applies the segmentation loss function to multiple levels of predictions to
    provide auxiliary losses for intermediate outputs in addition to the final
    output, used in FCN.
    """

    def __init__(
        self,
        reducer: LossReducer = mean_loss,
        feature_idx: tuple[int, ...] = (0,),
        loss_fn: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ] = seg_cross_entropy,
        weights: list[float] | None = None,
    ) -> None:
        """Creates an instance of the class.

        Args:
            reducer (LossReducer): Reducer for the loss function. Defaults to
                mean_loss.
            feature_idx (tuple[int]): Indices for the level of features to
                compute losses. Defaults to (0,).
            loss_fn (Callable, optional): Loss function that computes between
                predictions and targets. Defaults to seg_cross_entropy.
            weights (list[float], optional): The weights of each feature level.
                If None passes, it will set to 1 for all levels. Defaults to
                    None.
        """
        super().__init__(reducer)
        self.feature_idx = feature_idx
        self.loss_fn = loss_fn
        if weights is None:
            self.weights = [1.0] * len(self.feature_idx)
        else:
            self.weights = weights

    def forward(
        self, outputs: list[torch.Tensor], target: torch.Tensor
    ) -> LossesType:
        """Forward pass.

        Args:
            outputs (list[torch.Tensor]): Multi-level outputs.
            target (torch.Tensor): Assigned segmentation target mask.

        Returns:
            LossesType: Computed losses for each level.
        """
        losses: LossesType = {}
        for i, idx in enumerate(self.feature_idx):
            loss = self.reducer(self.loss_fn(outputs[idx], target))
            losses[f"loss_seg_level{idx}"] = torch.mul(self.weights[i], loss)

        return losses
