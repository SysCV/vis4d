"""CSP-Darknet base network used in YOLOX.

Modified from mmdetection (https://github.com/open-mmlab/mmdetection).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from vis4d.op.layer import Conv2d, CSPLayer


class Focus(nn.Module):
    """Focus width and height information into channel space.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        kernel_size (int, optional): The kernel size of the convolution.
            Defaults to 1.
        stride (int, optional): The stride of the convolution. Defaults to 1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
    ):
        """Init."""
        super().__init__()
        self.conv = Conv2d(
            in_channels * 4,
            out_channels,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2,
            bias=False,
            norm=nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03),
            activation=nn.SiLU(inplace=True),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            features (torch.Tensor): The input tensor of shape [B, C, W, H].
        """
        patch_top_left = features[..., ::2, ::2]
        patch_top_right = features[..., ::2, 1::2]
        patch_bot_left = features[..., 1::2, ::2]
        patch_bot_right = features[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        kernel_sizes (Sequence[int], optional): Sequential of kernel sizes of
            pooling layers. Defaults to (5, 9, 13).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: Sequence[int] = (5, 9, 13),
    ):
        """Init."""
        super().__init__()
        mid_channels = in_channels // 2
        self.conv1 = Conv2d(
            in_channels,
            mid_channels,
            1,
            stride=1,
            bias=False,
            norm=nn.BatchNorm2d(mid_channels, eps=0.001, momentum=0.03),
            activation=nn.SiLU(inplace=True),
        )
        self.poolings = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = mid_channels * (len(kernel_sizes) + 1)
        self.conv2 = Conv2d(
            conv2_channels,
            out_channels,
            1,
            bias=False,
            norm=nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03),
            activation=nn.SiLU(inplace=True),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            features (torch.Tensor): Input features.
        """
        x = self.conv1(features)
        x = torch.cat([x] + [pooling(x) for pooling in self.poolings], dim=1)
        x = self.conv2(x)
        return x


class CSPDarknet(nn.Module):
    """CSP-Darknet backbone used in YOLOv5 and YOLOX.

    Args:
        arch (str): Architecture of CSP-Darknet, from {P5, P6}.
            Default: P5.
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Default: 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Default: 1.0.
        out_indices (Sequence[int]): Output from which stages.
            Default: (2, 3, 4).
        frozen_stages (int): Stages to be frozen (stop grad and set eval
            mode). -1 means not freezing any parameters. Default: -1.
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Default: False.
        arch_ovewrite(list[list[int]], optional): Overwrite default arch
            settings. Defaults to None.
        spp_kernal_sizes: (tuple[int]): Sequential of kernel sizes of SPP
            layers. Default: (5, 9, 13).
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.

    Example:
        >>> import torch
        >>> from vis4d.op.base import CSPDarknet
        >>> self = CSPDarknet()
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 416, 416)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        ...
        (1, 256, 52, 52)
        (1, 512, 26, 26)
        (1, 1024, 13, 13)
    """

    # From left to right:
    # in_channels, out_channels, num_blocks, add_identity, use_spp
    arch_settings = {
        "P5": [
            [64, 128, 3, True, False],
            [128, 256, 9, True, False],
            [256, 512, 9, True, False],
            [512, 1024, 3, False, True],
        ],
        "P6": [
            [64, 128, 3, True, False],
            [128, 256, 9, True, False],
            [256, 512, 9, True, False],
            [512, 768, 3, True, False],
            [768, 1024, 3, False, True],
        ],
    }

    def __init__(
        self,
        arch: str = "P5",
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        out_indices: Sequence[int] = (2, 3, 4),
        frozen_stages: int = -1,
        arch_ovewrite: list[list[int]] | None = None,
        spp_kernal_sizes: Sequence[int] = (5, 9, 13),
        norm_eval: bool = False,
    ):
        """Init."""
        super().__init__()
        arch_setting = self.arch_settings[arch]
        if arch_ovewrite:
            arch_setting = arch_ovewrite
        assert set(out_indices).issubset(
            i for i in range(len(arch_setting) + 1)
        )
        if frozen_stages not in range(-1, len(arch_setting) + 1):
            raise ValueError(
                "frozen_stages must be in range(-1, "
                "len(arch_setting) + 1). But received "
                f"{frozen_stages}"
            )

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

        self.stem = Focus(
            3, int(arch_setting[0][0] * widen_factor), kernel_size=3
        )
        self.layers = ["stem"]

        for i, (
            in_channels,
            out_channels,
            num_blocks,
            add_identity,
            use_spp,
        ) in enumerate(arch_setting):
            in_channels = int(in_channels * widen_factor)
            out_channels = int(out_channels * widen_factor)
            num_blocks = max(round(num_blocks * deepen_factor), 1)
            stage: list[nn.Module] = []
            conv_layer = Conv2d(
                in_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                bias=False,
                norm=nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03),
                activation=nn.SiLU(inplace=True),
            )
            stage.append(conv_layer)
            if use_spp:
                spp = SPPBottleneck(
                    out_channels, out_channels, kernel_sizes=spp_kernal_sizes
                )
                stage.append(spp)
            csp_layer = CSPLayer(
                out_channels,
                out_channels,
                num_blocks=num_blocks,
                add_identity=bool(add_identity),
            )
            stage.append(csp_layer)
            self.add_module(f"stage{i + 1}", nn.Sequential(*stage))
            self.layers.append(f"stage{i + 1}")
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

    def _freeze_stages(self) -> None:
        """Freeze stages."""
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                m = getattr(self, self.layers[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode: bool = True) -> CSPDarknet:
        """Override the train mode for the model.

        Args:
            mode (bool): Whether to set training mode to True.
        """
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
        return self

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass.

        Args:
            images (torch.Tensor): Input images.
        """
        outs = [images, images]
        x = images
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return outs
