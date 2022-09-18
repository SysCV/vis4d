"""FCN Head for semantic segmentation."""

from typing import List, Optional, Tuple, NamedTuple, Callable

import torch
from torch import nn
import torch.nn.functional as F

from .base import BaseSegmentor
from .common import resize_feat


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


class FCN(_FCNBase):
    """FCN segmentation model.

    This is based on the implementation of
    `Fully Convolution Networks for Semantic Segmentation
    <https://arxiv.org/abs/1411.4038>`_.
    """

    def __init__(
        self,
        *args,
        kernel_sizes: List[int] = [4, 4, 16],
        strides: List[int] = [2, 2, 8],
        resize: Tuple[int, int] = (512, 512),
    ) -> None:
        """Init.

        Args:
            in_channels (List[int]): Number of channels in multi-level image
                feature.
            channels (int): Number of output channels. Usually the number of
                classes.
            kernel_sizes (List[int]): List of kernel size used in transposed
                convolutions. Its length must be same as the length of
                `in_channels`. Defaults to [4, 4, 16]
            strides (List[int]): List of stride used in transposed
                convolutions. Its length must be same as the length of
                `in_channels`. Defaults to [2, 2, 8]
            resize (Tuple[int, int]): The shape that prediction maps will be
                resized to. Defaults to (512, 512).
        """
        super().__init__(*args)
        self.size = resize
        self.num_used_channels = len(kernel_sizes)
        assert len(self.in_channels) == len(
            kernel_sizes
        ), "length of in_channels does not match the length of kernel_sizes."
        assert len(strides) == len(
            kernel_sizes
        ), "length of strides does not match the length of kernel_sizes."

        self.convs = nn.ModuleList()
        for i in range(self.num_used_channels):
            self.convs.append(
                nn.Conv2d(
                    self.in_channels[-i - 1],
                    self.out_channels,
                    kernel_size=1,
                )
            )

        self.transpose_convs = nn.ModuleList()
        for i in range(self.num_used_channels):
            self.transpose_convs.append(
                nn.ConvTranspose2d(
                    self.out_channels,
                    self.out_channels,
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    padding=1,
                )
            )

    def forward(self, x: List[torch.Tensor]) -> FCNOut:
        """Forward function for transforming feature maps and obtain
        segmentation prediction.

        Args:
            x (List[torch.Tensor]): List of multi-level image features.

        Returns:
            output (List[torch.Tensor]): Each tensor has shape (batch_size,
            self.channels, H, W) which is prediction for each FCN stages. E.g.,

            outputs[:-3] == x[:-3]
            outputs[-1] ==> "FCN 32s" output map
            outputs[-2] ==> "FCN 16s" output map
            outputs[-3] ==> "FCN 8s" output map
        """
        outputs = x.copy()
        combined_feat = self.convs[0](x[-1])
        outputs[-1] = resize_feat(combined_feat, self.size)

        for i in range(self.num_used_channels - 1):
            feat = x[-2 - i]
            combined_feat = self.transpose_convs[i](
                combined_feat
            ) + self.convs[i + 1](feat)
            outputs[-2 - i] = resize_feat(combined_feat, self.size)
        return FCNOut(pred=outputs[-3], outputs=outputs)


class FCNForResNet(_FCNBase):
    """FCN with ResNet backbone.

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
                output = resize_feat(output, self.resize)
            outputs[idx] = F.softmax(output, dim=1)
        return FCNOut(pred=outputs[-1], outputs=outputs)


class FCNLoss(nn.Module):
    def __init__(
        self,
        feature_idx: List[int],
        loss_fn: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ] = nn.CrossEntropyLoss,
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
        self.loss_fn = loss_fn()
        self.weights = weights

    def forward(self, outputs: List[torch.Tensor], target: torch.Tensor):
        losses = []
        total_loss = 0
        for i, idx in enumerate(self.feature_idx):
            loss = self.loss_fn(outputs[idx], target)
            total_loss += self.weights[i] * loss
            losses.append(loss)
        return FCNLosses(total_loss=total_loss, losses=losses)
