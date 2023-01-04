"""FCN Head for semantic segmentation."""
from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn


class FCNOut(NamedTuple):
    """Output of the FCN prediction."""

    pred: torch.Tensor  # logits for final prediction, (N, C, H, W)
    outputs: list[torch.Tensor]  # transformed feature maps


class FCNLosses(NamedTuple):
    """Losses for FCN"""

    total_loss: torch.Tensor
    losses: list[torch.Tensor]


class FCNHead(nn.Module):
    """FCN Head made with ResNet backbone.

    This is based on the implementation in `torchvision
    <https://github.com/pytorch/vision/blob/torchvision/models/segmentation/
    fcn.py>`_.
    """

    def __init__(
        self,
        in_channels: list[int],
        out_channels: int,
        dropout_prob: float = 0.1,
        resize: tuple[int, int] | None = None,
    ) -> None:
        """Creates an instance of the class.

        Args:
            in_channels (list[int]): Number of channels in multi-level image
                feature.
            out_channels (int): Number of output channels. Usually the number
                of classes.
            dropout_prob (float, optional): Dropout probability. Defaults to
                0.1.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resize = resize
        self.heads = nn.ModuleList()
        for in_channel in self.in_channels:
            self.heads.append(
                self._make_head(in_channel, self.out_channels, dropout_prob)
            )

    def _make_head(
        self, in_channels: int, channels: int, dropout_prob: float
    ) -> nn.Module:
        """Generate FCN segmentation head.

        Args:
            in_channels (int): Input feature channels.
            channels (int): Output segmentation channels.
            dropout_prob (float): Dropout probability.

        Returns:
            nn.Module: FCN segmentation head.
        """
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(
                in_channels,
                inter_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Conv2d(inter_channels, channels, kernel_size=1),
        ]
        return nn.Sequential(*layers)

    def forward(self, feats: list[torch.Tensor]) -> FCNOut:
        """Forward function for transforming feature maps and obtain
        segmentation prediction.

        Args:
            feats (list[torch.Tensor]): List of multi-level image features.

        Returns:
            output (list[torch.Tensor]): Each tensor has shape (batch_size,
            self.channels, H, W) which is prediction for each FCN stages. E.g.,

            outputs[-1] ==> main output map
            outputs[-2] ==> aux output map (e.g., used for training)
            outputs[:-2] ==> x[:-2]
        """
        outputs = feats.copy()
        num_features = len(feats)
        for i in range(len(self.in_channels)):
            idx = num_features - len(self.in_channels) + i
            feat = feats[idx]
            output = self.heads[i](feat)
            if self.resize:
                output = F.interpolate(
                    output,
                    size=self.resize,
                    mode="bilinear",
                    align_corners=False,
                )
            outputs[idx] = F.log_softmax(output, dim=1)
        return FCNOut(pred=outputs[-1], outputs=outputs)

    def __call__(self, feats: list[torch.Tensor]) -> FCNOut:
        """Type definition for function call."""
        return super()._call_impl(feats)


class FCNLoss(nn.Module):
    """FCN segmentation loss class."""

    def __init__(
        self,
        feature_idx: list[int],
        loss_fn: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ] = nn.CrossEntropyLoss(),
        weights: list[float] | None = None,
    ) -> None:
        """Creates an instance of the class.

        Args:
            feature_idx (list[int]): Indices for the level of features to
                compute losses.
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
    ) -> FCNLosses:
        """Forward pass.

        Args:
            outputs (list[torch.Tensor]): Multilevel FCN outputs.
            target (torch.Tensor): Assigned segmentation target mask.

        Returns:
            FCNLosses: computed losses for each level and the weighted total
                loss.
        """
        losses = []
        total_loss = torch.tensor(0.0, device=outputs[0].device)
        for i, idx in enumerate(self.feature_idx):
            loss = self.loss_fn(outputs[idx], target)
            total_loss += self.weights[i] * loss
            losses.append(loss)
        return FCNLosses(total_loss=total_loss, losses=losses)
