"""Wrapper for deformable convolution."""
import torch
from torch import nn

from vis4d.common.imports import MMCV_AVAILABLE

if MMCV_AVAILABLE:
    from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2dPack


BN_MOMENTUM = 0.1


class DeformConv(nn.Module):
    """Deformable Convolution."""

    def __init__(self, chi: int, cho: int) -> None:
        """Init."""
        assert MMCV_AVAILABLE, "DeformConv requires mmcv to be installed!"
        super().__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM), nn.ReLU(inplace=True)
        )
        self.conv = ModulatedDeformConv2dPack(
            chi,
            cho,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            dilation=1,
            deform_groups=1,
        )

    def forward(self, input_x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        input_x = self.conv(input_x)
        input_x = self.actf(input_x)
        return input_x
