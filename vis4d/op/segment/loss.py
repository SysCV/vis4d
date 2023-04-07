"""Semantic segmentation loss."""
from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from vis4d.common.typing import LossesType


class SegmentLoss(nn.Module):
    """Segmentation loss class."""

    def __init__(
        self,
        feature_idx: tuple[int, ...] = (0,),
        loss_fn: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ] = nn.CrossEntropyLoss(),
        weights: list[float] | None = None,
    ) -> None:
        """Creates an instance of the class.

        Args:
            feature_idx (tuple[int]): Indices for the level of features to
                compute losses. Defaults to (0,).
            loss_fn (Callable, optional): Loss function that computes between
                predictions and targets. Defaults to nn.NLLLoss.
            weights (list[float], optional): The weights of each feature level.
                If None passes, it will set to 1 for all levels. Defaults to
                    None.
        """
        super().__init__()
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
            outputs (list[torch.Tensor]): Multilevel outputs.
            target (torch.Tensor): Assigned segmentation target mask.

        Returns:
            LossesType: computed losses for each level and the weighted total
                loss.
        """
        tgt_h, tgt_w = target.shape[-2:]
        losses: LossesType = {}
        for i, idx in enumerate(self.feature_idx):
            loss = self.loss_fn(outputs[idx][:, :, :tgt_h, :tgt_w], target)
            losses[f"level_{idx}"] = self.weights[i] * loss
        return losses
