"""DLA-UP."""
import math
from typing import List, Optional

import numpy as np
import torch
from torch import nn

from vis4d.common.layers.conv2d import Conv2d, DeformConv
from vis4d.struct import FeatureMaps, NDArrayI64

from .base import BaseNeck, BaseNeckConfig


def fill_up_weights(up: nn.ConvTranspose2d) -> None:
    """Initialize weights of upsample layer."""
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (
                1 - math.fabs(j / f - c)
            )
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class IDAUp(nn.Module):  # type: ignore
    """IDAUp."""

    def __init__(
        self, use_dc: bool, o: int, channels: List[int], up_f: List[int]
    ) -> None:
        """Init."""
        super().__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            if use_dc:
                proj = DeformConv(c, o)
                node = DeformConv(o, o)
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
        self, layers: List[torch.Tensor], startp: int, endp: int
    ) -> None:
        """Forward."""
        for i in range(startp + 1, endp):
            upsample = getattr(self, "up_" + str(i - startp))
            project = getattr(self, "proj_" + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, "node_" + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])


class DLAUpConfig(BaseNeckConfig):
    """Config for DLAUp."""

    in_channels: List[int]
    out_channels: Optional[int]
    start_level: int = 0
    end_level: int = -1
    output_names: Optional[List[str]]
    use_deformable_convs: bool = True


class DLAUp(BaseNeck):
    """DLAUp."""

    def __init__(self, cfg: BaseNeckConfig) -> None:
        """Init."""
        super().__init__()
        self.cfg: DLAUpConfig = DLAUpConfig(**cfg.dict())
        self.start_level = self.cfg.start_level
        self.end_level = self.cfg.end_level
        if self.end_level == -1:
            self.end_level = len(self.cfg.in_channels)
        in_channels = self.cfg.in_channels[self.start_level : self.end_level]
        channels = list(in_channels)
        scales: NDArrayI64 = np.array(
            [2 ** i for i, _ in enumerate(in_channels)], dtype=np.int64
        )
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(
                self,
                f"ida_{i}",
                IDAUp(
                    self.cfg.use_deformable_convs,
                    channels[j],
                    in_channels[j:],
                    scales[j:] // scales[j],
                ),
            )
            scales[j + 1 :] = scales[j]
            in_channels[j + 1 :] = [channels[j] for _ in channels[j + 1 :]]
        out_channels = self.cfg.out_channels
        if out_channels is None:
            out_channels = channels[0]
        self.ida_final = IDAUp(
            self.cfg.use_deformable_convs,
            out_channels,
            channels,
            [2 ** i for i in range(self.end_level - self.start_level)],
        )

    def __call__(  # type: ignore[override]
        self,
        inputs: FeatureMaps,
    ) -> FeatureMaps:
        """Forward."""
        layers = list(inputs.values())
        outs = [layers[self.end_level - 1]]
        for i in range(self.end_level - self.start_level - 1):
            ida = getattr(self, f"ida_{i}")
            ida(layers, self.end_level - i - 2, self.end_level)
            outs.insert(0, layers[self.end_level - 1])
        self.ida_final(outs, 0, len(outs))
        outs = [outs[-1]]
        if self.cfg.output_names is None:  # pragma: no cover
            return {f"out{i}": v for i, v in enumerate(outs)}
        return dict(zip(self.cfg.output_names, outs))
