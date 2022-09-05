"""FCN Head for semantic segmentation.

This is based on the implementation of
`Fully Convolution Networks for Semantic Segmentation
<https://arxiv.org/abs/1411.4038>`_.
"""

from typing import List, Union

import torch
from torch import nn

from .base import BaseSegmentHead


class FCNHead(BaseSegmentHead):
    """FCN segmentation head."""

    def __init__(
        self,
        in_channels: Union[int, List[int]],
        channels: int,
        num_classes: int,
        num_convs: int = 2,
        kernel_size: int = 3,
        concat_input: bool = True,
        dilation: int = 1,
    ) -> None:
        """Init.

        Args:
            in_channels (List[int])
            channels (int)
            out_channel (int): Number of channels in the prediction, usually
                be the num of classes.
            num_convs (int): Number of convs in the head. Default: 2.
            kernel_size (int): The kernel size for convs in the head.
                Default: 3.
            concat_input (bool): Whether concat the input and output of convs
                before classification layer.
            dilation (int): The dilation rate for convs in the head.
                Default: 1.
        """
        super().__init__()
        if isinstance(in_channels, list):
            self.in_channels = sum(in_channels)
        else:
            self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        conv_padding = (kernel_size // 2) * dilation

        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            convs = []
            convs.append(
                nn.Conv2d(
                    self.in_channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                )
            )
            for _ in range(num_convs - 1):
                convs.append(
                    nn.Conv2d(
                        self.channels,
                        self.channels,
                        kernel_size=kernel_size,
                        padding=conv_padding,
                        dilation=dilation,
                    )
                )
            self.convs = nn.Sequential(*convs)
        if concat_input:
            self.conv_cat = nn.Conv2d(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
            )
        self.conv_seg = nn.Conv2d(self.channels, num_classes, kernel_size=1)

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        """Forward function for transforming feature maps and obtain
        segmentation prediction.

        Args:
            x (List[torch.Tensor]): List of multi-level image features.
        Returns:
            output (torch.Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        x = torch.cat(x, dim=1)
        feats = self.convs(x)
        if self.concat_input:
            feats = self.conv_cat(torch.cat([x, feats], dim=1))
        output = self.conv_seg(feats)
        return output
