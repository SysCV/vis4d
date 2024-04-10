"""YOLOX PAFPN.

Modified from mmdetection (https://github.com/open-mmlab/mmdetection).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from vis4d.op.layer import Conv2d, CSPLayer

from .base import FeaturePyramidProcessing


class YOLOXPAFPN(FeaturePyramidProcessing):
    """Path Aggregation Network used in YOLOX.

    Args:
        in_channels (list[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_csp_blocks (int, optional): Number of bottlenecks in CSPLayer.
            Defaults to 3.
        start_index (int, optional): Index of the first input feature map.
            Defaults to 2.
    """

    def __init__(
        self,
        in_channels: list[int],
        out_channels: int,
        num_csp_blocks: int = 3,
        start_index: int = 2,
    ):
        """Init."""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.start_index = start_index

        # build top-down blocks
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.reduce_layers.append(
                Conv2d(
                    in_channels[idx],
                    in_channels[idx - 1],
                    1,
                    bias=False,
                    norm=nn.BatchNorm2d(
                        in_channels[idx - 1], eps=0.001, momentum=0.03
                    ),
                    activation=nn.SiLU(inplace=True),
                )
            )
            self.top_down_blocks.append(
                CSPLayer(
                    in_channels[idx - 1] * 2,
                    in_channels[idx - 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                )
            )

        # build bottom-up blocks
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsamples.append(
                Conv2d(
                    in_channels[idx],
                    in_channels[idx],
                    3,
                    stride=2,
                    padding=1,
                    bias=False,
                    norm=nn.BatchNorm2d(
                        in_channels[idx], eps=0.001, momentum=0.03
                    ),
                    activation=nn.SiLU(inplace=True),
                )
            )
            self.bottom_up_blocks.append(
                CSPLayer(
                    in_channels[idx] * 2,
                    in_channels[idx + 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                )
            )

        self.out_convs = nn.ModuleList()
        for _, inc in enumerate(in_channels):
            self.out_convs.append(
                Conv2d(
                    inc,
                    out_channels,
                    1,
                    bias=False,
                    norm=nn.BatchNorm2d(
                        out_channels, eps=0.001, momentum=0.03
                    ),
                    activation=nn.SiLU(inplace=True),
                )
            )
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(
                    m.weight,
                    a=math.sqrt(5),
                    mode="fan_in",
                    nonlinearity="leaky_relu",
                )

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Forward pass.

        Args:
            features (tuple[Tensor]): Input features.

        Returns:
            list[Tensor]: YOLOXPAFPN features.
        """
        images, features = (
            features[: self.start_index],
            features[self.start_index :],
        )
        assert len(features) == len(self.in_channels)

        # top-down path
        inner_outs = [features[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = features[idx - 1]
            feat_heigh = self.reduce_layers[len(self.in_channels) - 1 - idx](
                feat_heigh
            )
            inner_outs[0] = feat_heigh

            upsample_feat = self.upsample(feat_heigh)

            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], 1)
            )
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](
                torch.cat([downsample_feat, feat_height], 1)
            )
            outs.append(out)

        # out convs
        for idx, conv in enumerate(self.out_convs):
            outs[idx] = conv(outs[idx])

        return images + outs

    def __call__(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Type definition for call implementation."""
        return self._call_impl(features)
