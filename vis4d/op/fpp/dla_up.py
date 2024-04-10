"""DLA-UP.

TODO(fyu) need clean up and update to the latest interface.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from vis4d.common import NDArrayI64

from ..layer import Conv2d, DeformConv
from .base import FeaturePyramidProcessing


def fill_up_weights(up_layer: nn.ConvTranspose2d) -> None:
    """Initialize weights of upsample layer."""
    w = up_layer.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (
                1 - math.fabs(j / f - c)
            )
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class IDAUp(nn.Module):
    """IDAUp."""

    def __init__(
        self, use_dc: bool, o: int, channels: list[int], up_f: list[int]
    ) -> None:
        """Creates an instance of the class."""
        super().__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            if use_dc:
                proj: Conv2d | DeformConv = DeformConv(
                    c,
                    o,
                    kernel_size=3,
                    padding=1,
                    norm=nn.BatchNorm2d(o),
                    activation=nn.ReLU(inplace=True),
                )
                node: Conv2d | DeformConv = DeformConv(
                    o,
                    o,
                    kernel_size=3,
                    padding=1,
                    norm=nn.BatchNorm2d(o),
                    activation=nn.ReLU(inplace=True),
                )
            else:
                proj = Conv2d(
                    c,
                    o,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                    norm=nn.BatchNorm2d(o),
                    activation=nn.ReLU(inplace=True),
                )
                node = Conv2d(
                    o,
                    o,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                    norm=nn.BatchNorm2d(o),
                    activation=nn.ReLU(inplace=True),
                )

            up = nn.ConvTranspose2d(
                o,
                o,
                f * 2,
                stride=f,
                padding=f // 2,
                output_padding=0,
                groups=o,
                bias=False,
            )
            fill_up_weights(up)

            setattr(self, "proj_" + str(i), proj)
            setattr(self, "up_" + str(i), up)
            setattr(self, "node_" + str(i), node)

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2.0 / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(
        self, layers: list[torch.Tensor], startp: int, endp: int
    ) -> None:
        """Forward."""
        for i in range(startp + 1, endp):
            upsample = getattr(self, "up_" + str(i - startp))
            project = getattr(self, "proj_" + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, "node_" + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])


class DLAUp(FeaturePyramidProcessing):
    """DLAUp."""

    def __init__(
        self,
        in_channels: list[int],
        out_channels: None | int = None,
        start_level: int = 0,
        end_level: int = -1,
        use_deformable_convs: bool = True,
    ) -> None:
        """Creates an instance of the class."""
        super().__init__()
        self.start_level = start_level
        self.end_level = end_level
        if self.end_level == -1:
            self.end_level = len(in_channels)
        in_channels = in_channels[self.start_level : self.end_level]
        channels = list(in_channels)
        scales: NDArrayI64 = np.array(
            [2**i for i, _ in enumerate(in_channels)], dtype=np.int64
        )
        for i in range(len(channels) - 1):
            j = -i - 2
            idaup = IDAUp(
                use_deformable_convs,
                channels[j],
                in_channels[j:],
                scales[j:] // scales[j],
            )
            setattr(self, f"ida_{i}", idaup)
            scales[j + 1 :] = scales[j]
            in_channels[j + 1 :] = [channels[j] for _ in channels[j + 1 :]]
        if out_channels is None:
            out_channels = channels[0]
        self.ida_final = IDAUp(
            use_deformable_convs,
            out_channels,
            channels,
            [2**i for i in range(self.end_level - self.start_level)],
        )

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Forward."""
        outs = [features[self.end_level - 1]]
        for i in range(self.end_level - self.start_level - 1):
            ida = getattr(self, f"ida_{i}")
            ida(features, self.end_level - i - 2, self.end_level)
            outs.insert(0, features[self.end_level - 1])
        self.ida_final(outs, 0, len(outs))
        outs = [outs[-1]]
        return outs
