"""Multi-level segmentation loss."""

from __future__ import annotations

import torch
from torch import Tensor

from vis4d.common.typing import LossesType

from .base import Loss
from .cross_entropy import cross_entropy
from .reducer import LossReducer, mean_loss


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
        weights: list[float] | None = None,
    ) -> None:
        """Creates an instance of the class.

        Args:
            reducer (LossReducer): Reducer for the loss function. Defaults to
                mean_loss.
            feature_idx (tuple[int]): Indices for the level of features to
                compute losses. Defaults to (0,).
            weights (list[float], optional): The weights of each feature level.
                If None passes, it will set to 1 for all levels. Defaults to
                    None.
        """
        super().__init__(reducer)
        self.feature_idx = feature_idx
        if weights is None:
            self.weights = [1.0] * len(self.feature_idx)
        else:
            self.weights = weights

    def forward(
        self, outputs: list[Tensor], target: Tensor, ignore_index: int = 255
    ) -> LossesType:
        """Forward pass.

        Args:
            outputs (list[Tensor]): Multi-level outputs.
            target (Tensor): Assigned segmentation target mask.
            ignore_index (int): Ignore class id. Default to 255.

        Returns:
            LossesType: Computed losses for each level.
        """
        losses: LossesType = {}
        tgt_h, tgt_w = target.shape[-2:]
        for i, idx in enumerate(self.feature_idx):
            loss = self.reducer(
                cross_entropy(
                    outputs[idx][:, :, :tgt_h, :tgt_w],
                    target,
                    ignore_index=ignore_index,
                )
            )
            losses[f"loss_seg_level{idx}"] = torch.mul(self.weights[i], loss)

        return losses
