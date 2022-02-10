"""Wrapper for deformable convolution."""
import torch
from torch import nn

from vis4d.common import Vis4DModule

try:
    from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2dPack

    MMCV_INSTALLED = True
except (ImportError, NameError):  # pragma: no cover
    MMCV_INSTALLED = False


BN_MOMENTUM = 0.1


class DeformConv(Vis4DModule[torch.Tensor, torch.Tensor]):
    """Deformable Convolution."""

    def __init__(self, chi: int, cho: int) -> None:
        """Init."""
        assert MMCV_INSTALLED, "DeformConv requires mmcv to be installed!"
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

    def __call__(self, input_x: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Forward."""
        input_x = self.conv(input_x)
        input_x = self.actf(input_x)
        return input_x
