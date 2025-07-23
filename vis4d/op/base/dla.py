"""DLA base model."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from vis4d.common.ckpt import load_model_checkpoint

from .base import BaseModel

BN_MOMENTUM = 0.1

DLA_MODEL_PREFIX = "http://dl.yf.io/dla/models/imagenet"

DLA_MODEL_MAPPING = {
    "dla34": "dla34-ba72cf86.pth",
    "dla46_c": "dla46_c-2bfd52c3.pth",
    "dla46x_c": "dla46x_c-d761bae7.pth",
    "dla60x_c": "dla60x_c-b870c45c.pth",
    "dla60": "dla60-24839fc4.pth",
    "dla60x": "dla60x-d15cacda.pth",
    "dla102": "dla102-d94d9790.pth",
    "dla102x": "dla102x-ad62be81.pth",
    "dla102x2": "dla102x2-262837b6.pth",
    "dla169": "dla169-0914e092.pth",
}

DLA_ARCH_SETTINGS = {  # pylint: disable=consider-using-namedtuple-or-dataclass
    "dla34": (
        (1, 1, 1, 2, 2, 1),
        (16, 32, 64, 128, 256, 512),
        False,
        "BasicBlock",
    ),
    "dla46_c": (
        (1, 1, 1, 2, 2, 1),
        (16, 32, 64, 64, 128, 256),
        False,
        "Bottleneck",
    ),
    "dla46x_c": (
        (1, 1, 1, 2, 2, 1),
        (16, 32, 64, 64, 128, 256),
        False,
        "BottleneckX",
    ),
    "dla60x_c": (
        (1, 1, 1, 2, 3, 1),
        (16, 32, 64, 64, 128, 256),
        False,
        "BottleneckX",
    ),
    "dla60": (
        (1, 1, 1, 2, 3, 1),
        (16, 32, 128, 256, 512, 1024),
        False,
        "Bottleneck",
    ),
    "dla60x": (
        (1, 1, 1, 2, 3, 1),
        (16, 32, 128, 256, 512, 1024),
        False,
        "BottleneckX",
    ),
    "dla102": (
        (1, 1, 1, 3, 4, 1),
        (16, 32, 128, 256, 512, 1024),
        True,
        "Bottleneck",
    ),
    "dla102x": (
        (1, 1, 1, 3, 4, 1),
        (16, 32, 128, 256, 512, 1024),
        True,
        "BottleneckX",
    ),
    "dla102x2": (
        (1, 1, 1, 3, 4, 1),
        (16, 32, 128, 256, 512, 1024),
        True,
        "BottleneckX",
    ),
    "dla169": (
        (1, 1, 2, 3, 5, 1),
        (16, 32, 128, 256, 512, 1024),
        True,
        "Bottleneck",
    ),
}


class BasicBlock(nn.Module):
    """BasicBlock."""

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        with_cp: bool = False,
    ) -> None:
        """Creates an instance of the class."""
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.stride = stride
        self.with_cp = with_cp

    def forward(
        self, input_x: Tensor, residual: None | Tensor = None
    ) -> Tensor:
        """Forward."""

        def _inner_forward(
            input_x: Tensor, residual: None | Tensor = None
        ) -> Tensor:
            if residual is None:
                residual = input_x
            out = self.conv1(input_x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            out += residual

            return out

        if self.with_cp and input_x.requires_grad:
            out = checkpoint(
                _inner_forward, input_x, residual, use_reentrant=True
            )
        else:
            out = _inner_forward(input_x, residual)

        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck."""

    expansion = 2

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        with_cp: bool = False,
    ) -> None:
        """Creates an instance of the class."""
        super().__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2d(
            inplanes, bottle_planes, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(
            bottle_planes,
            bottle_planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(
            bottle_planes, planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.with_cp = with_cp

    def forward(
        self, input_x: Tensor, residual: None | Tensor = None
    ) -> Tensor:
        """Forward."""

        def _inner_forward(
            input_x: Tensor, residual: None | Tensor = None
        ) -> Tensor:
            if residual is None:
                residual = input_x

            out = self.conv1(input_x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            out += residual

            return out

        if self.with_cp and input_x.requires_grad:
            out = checkpoint(
                _inner_forward, input_x, residual, use_reentrant=True
            )
        else:
            out = _inner_forward(input_x, residual)

        out = self.relu(out)

        return out


class BottleneckX(nn.Module):
    """BottleneckX."""

    expansion = 2
    cardinality = 32

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        with_cp: bool = False,
    ) -> None:
        """Creates an instance of the class."""
        super().__init__()
        cardinality = BottleneckX.cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(
            inplanes, bottle_planes, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(
            bottle_planes,
            bottle_planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation,
            groups=cardinality,
        )
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(
            bottle_planes, planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.with_cp = with_cp

    def forward(
        self, input_x: Tensor, residual: None | Tensor = None
    ) -> Tensor:
        """Forward."""

        def _inner_forward(
            input_x: Tensor, residual: None | Tensor = None
        ) -> Tensor:
            if residual is None:
                residual = input_x

            out = self.conv1(input_x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            out += residual

            return out

        if self.with_cp and input_x.requires_grad:
            out = checkpoint(
                _inner_forward, input_x, residual, use_reentrant=True
            )
        else:
            out = _inner_forward(input_x, residual)

        out = self.relu(out)

        return out


class Root(nn.Module):
    """Root."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        residual: bool,
        with_cp: bool = False,
    ) -> None:
        """Creates an instance of the class."""
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=1,
            bias=False,
            padding=(kernel_size - 1) // 2,
        )
        self.bn = nn.BatchNorm2d(  # pylint: disable=invalid-name
            out_channels, momentum=BN_MOMENTUM
        )
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual
        self.with_cp = with_cp

    def forward(self, *input_x: Tensor) -> Tensor:
        """Forward."""

        def _inner_forward(*input_x: Tensor) -> Tensor:
            feats = self.conv(torch.cat(input_x, 1))
            feats = self.bn(feats)
            if self.residual:
                feats += input_x[0]
            return feats

        if self.with_cp and input_x[0].requires_grad:
            feats = checkpoint(_inner_forward, *input_x, use_reentrant=True)
        else:
            feats = _inner_forward(*input_x)

        feats = self.relu(feats)

        return feats


class Tree(nn.Module):
    """Tree."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        levels: int,
        block: str,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        level_root: bool = False,
        root_dim: int = 0,
        root_kernel_size: int = 1,
        dilation: int = 1,
        root_residual: bool = False,
        with_cp: bool = False,
    ) -> None:
        """Creates an instance of the class."""
        super().__init__()
        if block == "BasicBlock":
            block_c = BasicBlock
        elif block == "Bottleneck":
            block_c = Bottleneck  # type: ignore
        elif block == "BottleneckX":
            block_c = BottleneckX  # type: ignore
        else:
            raise ValueError(f"Block={block} not yet supported in DLA!")
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1: Tree | BasicBlock = block_c(
                in_channels,
                out_channels,
                stride,
                dilation=dilation,
                with_cp=with_cp,
            )
            self.tree2: Tree | BasicBlock = block_c(
                out_channels,
                out_channels,
                1,
                dilation=dilation,
                with_cp=with_cp,
            )
            self.root = Root(
                root_dim,
                out_channels,
                root_kernel_size,
                root_residual,
                with_cp=with_cp,
            )
        else:
            self.tree1 = Tree(
                levels - 1,
                block,
                in_channels,
                out_channels,
                stride,
                root_dim=0,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual,
                with_cp=with_cp,
            )
            self.tree2 = Tree(
                levels - 1,
                block,
                out_channels,
                out_channels,
                root_dim=root_dim + out_channels,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual,
                with_cp=with_cp,
            )
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels and levels == 1:
            # NOTE the official impl/weights have project layers in levels > 1
            # case that are never used, hence 'levels == 1' is added but
            # pretrained models will need strict=False while loading.
            self.project = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(
        self,
        input_x: Tensor,
        residual: None | Tensor = None,
        children: None | list[Tensor] = None,
    ) -> Tensor:
        """Forward."""
        children = [] if children is None else children
        bottom = self.downsample(input_x) if self.downsample else input_x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        input_x1 = self.tree1(input_x, residual)
        if self.levels == 1:
            input_x2 = self.tree2(input_x1)
            input_x = self.root(input_x2, input_x1, *children)
        else:
            children.append(input_x1)
            input_x = self.tree2(input_x1, children=children)
        return input_x


class DLA(BaseModel):
    """DLA base model."""

    def __init__(
        self,
        name: str,
        out_indices: Sequence[int] = (0, 1, 2, 3),
        with_cp: bool = False,
        pretrained: bool = False,
        weights: None | str = None,
    ) -> None:
        """Creates an instance of the class."""
        super().__init__()
        assert name in DLA_ARCH_SETTINGS, f"{name} is not supported!"

        levels, channels, residual_root, block = DLA_ARCH_SETTINGS[name]

        if name == "dla102x2":  # pragma: no cover
            BottleneckX.cardinality = 64

        self.base_layer = nn.Sequential(
            nn.Conv2d(
                3, channels[0], kernel_size=7, stride=1, padding=3, bias=False
            ),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0]
        )
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2
        )
        self.level2 = Tree(
            levels[2],
            block,
            channels[1],
            channels[2],
            2,
            level_root=False,
            root_residual=residual_root,
            with_cp=with_cp,
        )
        self.level3 = Tree(
            levels[3],
            block,
            channels[2],
            channels[3],
            2,
            level_root=True,
            root_residual=residual_root,
            with_cp=with_cp,
        )
        self.level4 = Tree(
            levels[4],
            block,
            channels[3],
            channels[4],
            2,
            level_root=True,
            root_residual=residual_root,
            with_cp=with_cp,
        )
        self.level5 = Tree(
            levels[5],
            block,
            channels[4],
            channels[5],
            2,
            level_root=True,
            root_residual=residual_root,
            with_cp=with_cp,
        )

        self.out_indices = out_indices
        self._out_channels = [channels[i + 2] for i in out_indices]

        if pretrained:
            if weights is None:  # pragma: no cover
                weights = f"{DLA_MODEL_PREFIX}/{DLA_MODEL_MAPPING[name]}"

            load_model_checkpoint(self, weights)

        else:
            self._init_weights()

    def _init_weights(self) -> None:
        """Initialize module weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    @staticmethod
    def _make_conv_level(
        inplanes: int,
        planes: int,
        convs: int,
        stride: int = 1,
        dilation: int = 1,
    ) -> nn.Sequential:
        """Build convolutional level."""
        modules = []
        for i in range(convs):
            modules.extend(
                [
                    nn.Conv2d(
                        inplanes,
                        planes,
                        kernel_size=3,
                        stride=stride if i == 0 else 1,
                        padding=dilation,
                        bias=False,
                        dilation=dilation,
                    ),
                    nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True),
                ]
            )
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, images: Tensor) -> list[Tensor]:
        """DLA forward.

        Args:
            images (Tensor[N, C, H, W]): Image input to process. Expected to
                type float32 with values ranging 0..255.

        Returns:
            fp (list[Tensor]): The output feature pyramid. The list index
            represents the level, which has a downsampling raio of 2^index.
        """
        input_x = self.base_layer(images)

        outs = [images, images]

        for i in range(6):
            input_x = getattr(self, f"level{i}")(input_x)

            if i - 2 in self.out_indices:
                outs.append(input_x)

        return outs

    @property
    def out_channels(self) -> list[int]:
        """Get the numbers of channels for each level of feature pyramid.

        Returns:
            list[int]: number of channels
        """
        return [3, 3] + self._out_channels
