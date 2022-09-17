"""FCN Head for semantic segmentation."""

from typing import List, Optional, Tuple

import torch
from torch import nn

from .base import BaseSegmentor


class FCN(BaseSegmentor):
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
        resize: Optional[Tuple[int, int]] = None,
        align_corners: bool = False,
        **kwargs,
    ) -> None:
        """Init.

        Args:
            kernel_sizes (List[int]): List of kernel size used in transposed
                convolutions. Its length must be same as the length of
                `in_channels`. Defaults to [4, 4, 16]
            strides (List[int]): List of stride used in transposed
                convolutions. Its length must be same as the length of
                `in_channels`. Defaults to [2, 2, 8]
        """
        super().__init__(*args, **kwargs)
        self.num_features = len(self.in_channels)
        assert (
            len(kernel_sizes) == self.num_features
        ), "length of kernel_sizes does not match the number of features."
        assert (
            len(strides) == self.num_features
        ), "length of strides does not match the number of features."

        self.convs = nn.ModuleList()
        for i in range(self.num_features):
            self.convs.append(
                nn.Conv2d(
                    self.in_channels[self.num_features - i - 1],
                    self.channels,
                    kernel_size=1,
                )
            )

        self.transpose_convs = nn.ModuleList()
        for i in range(self.num_features):
            self.transpose_convs.append(
                nn.ConvTranspose2d(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    padding=1,
                )
            )

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward function for transforming feature maps and obtain
        segmentation prediction.

        Args:
            x (List[torch.Tensor]): List of multi-level image features.

        Returns:
            output (List[torch.Tensor]): Each tensor has shape (batch_size,
            self.channels, H, W) which is prediction for each FCN stages. E.g.,

            output[0] ==> "FCN 8s" output map
            output[1] ==> "FCN 16s" output map
            output[2] ==> "FCN 32s" output map
        """
        outputs = []
        combined_feat = self.convs[0](x[self.num_features - 1])
        outputs.append(self._upsample_feat(combined_feat, self.resize))

        for i in range(self.num_features - 1):
            feat = x[self.num_features - i - 2]
            combined_feat = self.transpose_convs[i](
                combined_feat
            ) + self.convs[i + 1](feat)
            outputs.append(self._upsample_feat(combined_feat, self.resize))
        outputs.reverse()
        return outputs


class FCNResNet(BaseSegmentor):
    """FCN with ResNet backbone.

    This is based on the implementation in `torchvision
    <https://github.com/pytorch/vision/blob/torchvision/models/segmentation/
    fcn.py>`_.
    """

    def __init__(
        self,
        *args,
        dropout_prob: float = 0.1,
        **kwargs,
    ) -> None:
        """Init.

        Args:
            dropout_prob (float, optional): Dropout probability. Defaults to
                0.1.
        """
        super().__init__(*args, **kwargs)
        self.heads = nn.ModuleList()
        for in_channels in self.in_channels:
            self.heads.append(
                self._get_head(in_channels, self.channels, dropout_prob)
            )

    def _get_head(self, in_channels: int, channels: int, dropout_prob: float):
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

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward function for transforming feature maps and obtain
        segmentation prediction.

        Args:
            x (List[torch.Tensor]): List of multi-level image features.

        Returns:
            output (List[torch.Tensor]): Each tensor has shape (batch_size,
            self.channels, H, W) which is prediction for each FCN stages. E.g.,

            output[0] ==> main output map
            output[1:] ==> aux output maps (e.g., used during training)
        """
        outputs = []
        for i, feat in enumerate(x):
            output = self.heads[i](feat)
            if self.resize:
                outputs = self._upsample_feat(output, self.resize)
            outputs.append(outputs)
        outputs.reverse()
        return outputs
