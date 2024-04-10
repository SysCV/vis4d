"""Unet Implementation based on https://arxiv.org/abs/1505.04597.

Code taken from https://github.com/jaxony/unet-pytorch/blob/master/model.py
and modified to include typing and custom ops.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from vis4d.op.layer.conv2d import UnetDownConv, UnetUpConv


class UNetOut(NamedTuple):
    """Output of the UNet operator.

    logits: Final output of the network without applying softmax
    intermediate_features: Intermediate features of the upsampling path
                            at different scales.
    """

    logits: torch.Tensor
    intermediate_features: list[torch.Tensor]


class UNet(nn.Module):
    """The U-Net is a convolutional encoder-decoder neural network.

    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).

    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        depth: int = 5,
        start_filts: int = 32,
        up_mode: str = "transpose",
        merge_mode: str = "concat",
    ):
        """Unet Operator.

        Args:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            num_classes: int, number of output classes.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
            merge_mode: string, how to merge features, can be 'concat' or 'add'


        Raises:
            ValueError: if invalid modes are provided
        """
        super().__init__()

        if up_mode in {"transpose", "upsample"}:
            self.up_mode = up_mode
        else:
            raise ValueError(
                f"{up_mode} is not a valid mode for  upsampling. Only"
                f"'transpose' and 'upsample' are allowed."
            )

        if merge_mode in {"concat", "add"}:
            self.merge_mode = merge_mode
        else:
            raise ValueError(
                f'"{up_mode}" is not a valid mode for'
                f"merging up and down paths. "
                f'Only "concat" and '
                f'"add" are allowed.'
            )

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == "upsample" and self.merge_mode == "add":
            raise ValueError(
                'up_mode "upsample" is incompatible '
                'with merge_mode "add" at the moment '
                "because it doesn't make sense to use "
                "nearest neighbour to reduce "
                "depth channels (by half)."
            )

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth

        self.down_convs: nn.ModuleList = nn.ModuleList()

        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs  # type: ignore
            outs = self.start_filts * (2**i)
            pooling = i < (depth - 1)

            down_conv = UnetDownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        self.up_convs: nn.ModuleList = nn.ModuleList()

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            up_conv = UnetUpConv(
                ins, outs, up_mode=up_mode, merge_mode=merge_mode
            )
            self.up_convs.append(up_conv)
        self.conv_final = nn.Conv2d(
            outs, num_classes, kernel_size=1, groups=1, stride=1
        )

    def __call__(self, data: torch.Tensor) -> UNetOut:
        """Applies the UNet.

        Args:
            data (tensor): Input Images into the network shape [N, C, W, H]

        """
        return self._call_impl(data)

    def forward(self, data: torch.Tensor) -> UNetOut:
        """Applies the UNet.

        Args:
            data (tensor): Input Images into the network shape [N, C, W, H]
        """
        encoder_outs: list[torch.Tensor] = []
        inter_feats: list[torch.Tensor] = []
        # encoder pathway, save outputs for merging

        for down_conv in self.down_convs:
            out = down_conv(data)
            data = out.pooled_features
            encoder_outs.append(out.features)

        for level, up_conv in enumerate(self.up_convs):
            before_pool = encoder_outs[-(level + 2)]
            data = up_conv(before_pool, data)
            inter_feats.append(data)

        logits = self.conv_final(data)
        return UNetOut(logits=logits, intermediate_features=inter_feats)
