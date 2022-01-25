"""DLA backbone."""
from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.utils import model_zoo

from vis4d.common.module import Vis4DModule
from vis4d.struct import ArgsType, FeatureMaps, InputSample

from .base import BaseBackbone

BN_MOMENTUM = 0.1
DLA_MODEL_PREFIX = "http://dl.yf.io/dla/models/"
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
DLA_ARCH_SETTINGS = {
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


class BasicBlock(Vis4DModule[torch.Tensor, torch.Tensor]):
    """BasicBlock."""

    def __init__(
        self, inplanes: int, planes: int, stride: int = 1, dilation: int = 1
    ) -> None:
        """Init."""
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

    def __call__(  # type: ignore
        self, input_x: torch.Tensor, residual: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward."""
        if residual is None:
            residual = input_x

        out = self.conv1(input_x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(Vis4DModule[torch.Tensor, torch.Tensor]):
    """Bottleneck."""

    expansion = 2

    def __init__(
        self, inplanes: int, planes: int, stride: int = 1, dilation: int = 1
    ) -> None:
        """Init."""
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

    def __call__(  # type: ignore
        self, input_x: torch.Tensor, residual: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward."""
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
        out = self.relu(out)

        return out


class BottleneckX(Vis4DModule[torch.Tensor, torch.Tensor]):
    """BottleneckX."""

    expansion = 2
    cardinality = 32

    def __init__(
        self, inplanes: int, planes: int, stride: int = 1, dilation: int = 1
    ) -> None:
        """Init."""
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

    def __call__(  # type: ignore
        self, input_x: torch.Tensor, residual: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward."""
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
        out = self.relu(out)

        return out


class Root(Vis4DModule[torch.Tensor, torch.Tensor]):
    """Root."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        residual: bool,
    ) -> None:
        """Init."""
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=1,
            bias=False,
            padding=(kernel_size - 1) // 2,
        )
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def __call__(self, *input_x: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Forward."""
        children = input_x
        input_x = self.conv(torch.cat(input_x, 1))
        input_x = self.bn(input_x)
        if self.residual:
            input_x += children[0]
        input_x = self.relu(input_x)

        return input_x


class Tree(Vis4DModule[torch.Tensor, torch.Tensor]):
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
    ) -> None:
        """Init."""
        super().__init__()
        if block == "BasicBlock":
            block_c = BasicBlock
        elif block == "Bottleneck":
            block_c = Bottleneck
        elif block == "BottleneckX":
            block_c = BottleneckX
        else:
            raise ValueError(f"Block={block} not yet supported in DLA!")
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block_c(
                in_channels, out_channels, stride, dilation=dilation
            )
            self.tree2 = block_c(
                out_channels, out_channels, 1, dilation=dilation
            )
            self.root = Root(
                root_dim, out_channels, root_kernel_size, root_residual
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
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
            )

    def __call__(  # type: ignore
        self,
        input_x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        children: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
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
            input_x = self.tree2(input_x1, children=children)  # type: ignore
        return input_x


class DLA(BaseBackbone):
    """DLA backbone."""

    def __init__(
        self,
        *args: ArgsType,
        name: Optional[str] = None,
        levels: Tuple[int, int, int, int, int, int] = (1, 1, 1, 2, 2, 1),
        channels: Tuple[int, int, int, int, int, int] = (
            16,
            32,
            64,
            128,
            256,
            512,
        ),
        block: str = "BasicBlock",
        residual_root: bool = False,
        cardinality: int = 32,
        weights: Optional[str] = None,
        style: str = "imagenet",
        **kwargs: ArgsType,
    ) -> None:
        """Init."""
        super().__init__(*args, **kwargs)
        if name is not None:
            assert name in DLA_ARCH_SETTINGS
            arch_setting = DLA_ARCH_SETTINGS[name]
            levels, channels, residual_root, block = arch_setting
            if name == "dla102x2":  # pragma: no cover
                BottleneckX.cardinality = 64
        else:
            BottleneckX.cardinality = cardinality
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
        )
        self.level3 = Tree(
            levels[3],
            block,
            channels[2],
            channels[3],
            2,
            level_root=True,
            root_residual=residual_root,
        )
        self.level4 = Tree(
            levels[4],
            block,
            channels[3],
            channels[4],
            2,
            level_root=True,
            root_residual=residual_root,
        )
        self.level5 = Tree(
            levels[5],
            block,
            channels[4],
            channels[5],
            2,
            level_root=True,
            root_residual=residual_root,
        )

        if weights is not None:  # pragma: no cover
            if weights.startswith("dla://"):
                weights_name = weights.split("dla://")[-1]
                assert weights_name in DLA_MODEL_MAPPING
                weights = (
                    f"{DLA_MODEL_PREFIX}{style}/"
                    f"{DLA_MODEL_MAPPING[weights_name]}"
                )
            self.load_pretrained_model(weights)

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

    def load_pretrained_model(self, weights: str) -> None:
        """Load pretrained weights."""
        if weights.startswith("http://") or weights.startswith("https://"):
            model_weights = model_zoo.load_url(weights)
        else:  # pragma: no cover
            model_weights = torch.load(weights)
        self.load_state_dict(model_weights, strict=False)

    def __call__(  # type: ignore[override]
        self, inputs: InputSample
    ) -> FeatureMaps:
        """Backbone forward."""
        inputs = self.preprocess_inputs(inputs)
        outs = []
        input_x = self.base_layer(inputs.images.tensor)
        for i in range(6):
            input_x = getattr(self, f"level{i}")(input_x)
            outs.append(input_x)
        backbone_outs = self.get_outputs(outs)
        if self.neck is not None:
            return self.neck(backbone_outs)
        return backbone_outs
