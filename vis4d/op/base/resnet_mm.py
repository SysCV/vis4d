"""Residual networks base model."""
from __future__ import annotations

from collections.abc import Callable, Sequence

from torch import nn, Tensor
import torch.utils.checkpoint as cp
from torch.nn.modules.batchnorm import _BatchNorm

from vis4d.op.layer.deform_conv import ModulatedDeformConv2dPack

from .base import BaseModel


def build_conv_layer(
    in_planes: int,
    out_planes: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    bias: bool = False,
    use_dcn: bool = False,
) -> nn.Module:
    """Build a convolution layer."""
    if use_dcn:
        return ModulatedDeformConv2dPack(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )


class BasicBlock(nn.Module):
    """BasicBlock."""

    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: nn.Module | None = None,
        style: str = "pytorch",
        with_cp: bool = False,
        with_dcn: bool = False,
        norm_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = build_conv_layer(
            inplanes,
            planes,
            3,
            stride=stride,
            dilation=dilation,
            padding=dilation,
            bias=False,
        )
        self.bn1 = norm_layer(planes)
        self.conv2 = build_conv_layer(planes, planes, 3, padding=1, bias=False)
        self.bn2 = norm_layer(planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck."""

    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: nn.Module | None = None,
        style: str = "pytorch",
        with_cp: bool = False,
        with_dcn: bool = False,
        norm_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        """Bottleneck block for ResNet.

        If style is "pytorch", the stride-two layer is the 3x3 conv layer, if
        it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

        assert style in ["pytorch", "caffe"]
        if style == "pytorch":
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.conv1 = build_conv_layer(
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False,
        )
        self.bn1 = norm_layer(planes)

        if not with_dcn:
            self.conv2 = build_conv_layer(
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False,
            )
        else:
            self.conv2 = build_conv_layer(
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False,
                use_dcn=True,
            )
        self.bn2 = norm_layer(planes)

        self.conv3 = build_conv_layer(
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


# TODO: Add stem, init
class ResNet(BaseModel):
    """ResNet BaseModel.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        stem_channels (int | None): Number of stem channels. If not specified,
            it will be the same as `base_channels`. Default: None.
        base_channels (int): Number of base channels of res layer. Default: 64.
        in_channels (int): Number of input image channels. Default: 3.
        num_stages (int): Resnet stages. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        plugins (list[dict]): List of plugins for stages, each dict contains:

            - cfg (dict, required): Cfg dict to build plugin.
            - position (str, required): Position inside block to insert
              plugin, options are 'after_conv1', 'after_conv2', 'after_conv3'.
            - stages (tuple[bool], optional): Stages to apply plugin, length
              should be same as 'num_stages'.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
        pretrained (str, optional): model pretrained path. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None

    Example:
        >>> from mmdet.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=18)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3)),
    }

    def __init__(
        self,
        depth: int,
        in_channels: int = 3,
        base_channels: int = 64,
        num_stages: int = 4,
        strides: Sequence[int] = (1, 2, 2, 2),
        dilations: Sequence[int] = (1, 1, 1, 1),
        out_indices: Sequence[int] = (0, 1, 2, 3),
        style: str = "pytorch",
        frozen_stages: int = -1,
        norm_eval: bool = True,
        stage_with_dcn: Sequence[bool] = (False, False, False, False),
        with_cp: bool = False,
        norm_layer: Callable[..., nn.Module] | None = None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        if depth not in self.arch_settings:
            raise KeyError(f"invalid depth {depth} for resnet")

        self.depth = depth
        stem_channels = base_channels

        assert num_stages >= 1 and num_stages <= 4
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages

        self.frozen_stages = frozen_stages

        self.with_cp = with_cp
        self.norm_eval = norm_eval

        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            planes = base_channels * 2**i
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=style,
                with_cp=with_cp,
                with_dcn=stage_with_dcn[i],
            )
            self.inplanes = planes * self.block.expansion
            layer_name = f"layer{i + 1}"
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = (
            self.block.expansion
            * base_channels
            * 2 ** (len(self.stage_blocks) - 1)
        )

    def make_res_layer(
        self,
        block: BasicBlock | Bottleneck,
        inplanes: int,
        planes: int,
        num_blocks: int,
        stride: int,
        dilation: int,
        style: str,
        with_cp: bool,
        with_dcn: bool,
        downsample_first: bool = True,
    ) -> nn.Sequential:
        """Pack all blocks in a stage into a ``ResLayer``."""
        layers = []
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = []
            conv_stride = stride
            downsample.extend(
                [
                    build_conv_layer(
                        inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=conv_stride,
                        bias=False,
                    ),
                    norm_layer(planes * block.expansion),
                ]
            )
            downsample = nn.Sequential(*downsample)

        layers = []
        if downsample_first:
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    dilation=dilation,
                    downsample=downsample,
                    style=style,
                    with_cp=with_cp,
                    with_dcn=with_dcn,
                    norm_layer=norm_layer,
                )
            )
            inplanes = planes * block.expansion
            for _ in range(1, num_blocks):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=planes,
                        stride=1,
                        dilation=dilation,
                        style=style,
                        with_cp=with_cp,
                        with_dcn=with_dcn,
                        norm_layer=norm_layer,
                    )
                )

        else:  # downsample_first=False is for HourglassModule
            for _ in range(num_blocks - 1):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=planes,
                        stride=1,
                        dilation=dilation,
                        style=style,
                        with_cp=with_cp,
                        with_dcn=with_dcn,
                        norm_layer=norm_layer,
                    )
                )
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    dilation=dilation,
                    downsample=downsample,
                    style=style,
                    with_cp=with_cp,
                    with_dcn=with_dcn,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)

    def _make_stem_layer(self, in_channels: int, stem_channels: int) -> None:
        """Make stem layer for ResNet."""
        self.conv1 = build_conv_layer(
            in_channels,
            stem_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = self._norm_layer(stem_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.bn1.eval()
            for m in [self.conv1, self.bn1]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f"layer{i}")
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, images: Tensor) -> list[Tensor]:
        """Forward function."""
        x = self.conv1(images)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # outs = [images, images]
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super().train(mode)
        self._freeze_stages()

        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    @property
    def out_channels(self) -> list[int]:
        """Get the number of channels for each level of feature pyramid.

        Returns:
            list[int]: number of channels
        """
        return [3, 3, 256, 512, 1024, 2048]
