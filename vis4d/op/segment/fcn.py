"""FCN Head for semantic segmentation."""

from typing import Callable, List, NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .base import BaseSegmentor


class FCNOut(NamedTuple):
    """Output of the FCN prediction."""

    pred: torch.Tensor  # logits for final prediction, (N, C, H, W)
    outputs: List[torch.Tensor]  # transformed feature maps


class FCNLosses(NamedTuple):
    """Losses for FCN"""

    total_loss: torch.Tensor
    losses: List[torch.Tensor]


class _FCNBase(BaseSegmentor):
    def __init__(self, in_channels: List[int], out_channels: int) -> None:
        """Init.

        Args:
            in_channels (List[int]): Number of channels in multi-level image
                feature.
            out_channels (int): Number of output channels. Usually the number
                of classes.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels


class FCNHead(_FCNBase):
    """FCN Head made with ResNet backbone.

    This is based on the implementation in `torchvision
    <https://github.com/pytorch/vision/blob/torchvision/models/segmentation/
    fcn.py>`_.
    """

    def __init__(
        self,
        *args,
        dropout_prob: float = 0.1,
        resize: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Init.

        Args:
            in_channels (List[int]): Number of channels in multi-level image
                feature.
            out_channels (int): Number of output channels. Usually the number of
                classes.
            seg_channel_idx (List[int]): Indices of channel that used to get
                segmentation maps. Defaults to [4, 5].
            dropout_prob (float, optional): Dropout probability. Defaults to
                0.1.
        """
        super().__init__(*args)
        self.resize = resize
        self.heads = nn.ModuleList()
        for idx in range(len(self.in_channels)):
            self.heads.append(
                self._make_head(
                    self.in_channels[idx], self.out_channels, dropout_prob
                )
            )

    def _make_head(self, in_channels: int, channels: int, dropout_prob: float):
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

    def forward(self, x: List[torch.Tensor]) -> FCNOut:
        """Forward function for transforming feature maps and obtain
        segmentation prediction.

        Args:
            x (List[torch.Tensor]): List of multi-level image features.

        Returns:
            output (List[torch.Tensor]): Each tensor has shape (batch_size,
            self.channels, H, W) which is prediction for each FCN stages. E.g.,

            outputs[-1] ==> main output map
            outputs[-2] ==> aux output map (e.g., used for training)
            outputs[:-2] ==> x[:-2]
        """
        outputs = x.copy()
        num_features = len(x)
        for i in range(len(self.in_channels)):
            idx = num_features - len(self.in_channels) + i
            feat = x[idx]
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


class FCNLoss(nn.Module):
    def __init__(
        self,
        feature_idx: List[int],
        loss_fn: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ] = nn.CrossEntropyLoss(),
        weights: List[float] = [0.5, 1],
    ) -> None:
        """Init.

        Args:
            feature_idx (List[int]): Indices for the level of features that
                contain segmentation results.
            loss_fn (Callable, optional): Loss function that computes between
                predictions and targets. Defaults to nn.NLLLoss.
            weights (List[float]):
        """

        super().__init__()
        self.feature_idx = feature_idx
        self.loss_fn = loss_fn
        self.weights = weights

    def forward(self, outputs: List[torch.Tensor], target: torch.Tensor):
        losses = []
        total_loss = 0
        for i, idx in enumerate(self.feature_idx):
            loss = self.loss_fn(outputs[idx], target)
            total_loss += self.weights[i] * loss
            losses.append(loss)
        return FCNLosses(total_loss=total_loss, losses=losses)
