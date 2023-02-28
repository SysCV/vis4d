"""Visual Transformer base model."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from vis4d.op.layer.attention import Attention2d
from vis4d.op.layer.conv2d import ConvBN2d
from vis4d.op.layer.drop import DropPath
from vis4d.op.layer.mlp import MLP

from .base import BaseModel


class _PatchEmbed(nn.Module):
    """Patch embedding layer."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        resolution: tuple[int, int],
        activation: nn.Module,
    ):
        """Generate embedding for each patch of images.

        Args:
            in_channels (int): Number of input channels.
            embed_dim (int): Number of output channels.
            resolution (tuple[int, int]): Input resolution.
            activation (nn.Module): Activation function.
        """
        super().__init__()
        img_size: tuple[int, int] = resolution
        self.patches_resolution = (img_size[0] // 4, img_size[1] // 4)
        self.num_patches = (
            self.patches_resolution[0] * self.patches_resolution[1]
        )
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.seq = nn.Sequential(
            ConvBN2d(
                in_channels, embed_dim // 2, kernel_size=3, stride=2, padding=1
            ),
            activation(),
            ConvBN2d(
                embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.seq(x)


class _PatchMerging(nn.Module):
    """Patch merging layer."""

    def __init__(
        self,
        input_resolution: tuple[int, int],
        dim: int,
        out_dim: int,
        activation: nn.Module,
    ) -> None:
        """Initialize the patch merging layer.

        Args:
            input_resolution (tuple[int, int]): Input resolution.
            dim (int): Number of input channels.
            out_dim (int): Number of output channels.
            activation (nn.Module): Activation function.
        """
        super().__init__()

        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim
        self.act = activation()
        self.conv1 = ConvBN2d(dim, out_dim, kernel_size=1, stride=1)
        self.conv2 = ConvBN2d(
            out_dim,
            out_dim,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=out_dim,
        )
        self.conv3 = ConvBN2d(out_dim, out_dim, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if x.ndim == 3:
            height, width = self.input_resolution
            num_batches = len(x)
            # (num_batches, num_channels, height, width)
            x = x.view(num_batches, height, width, -1).permute(0, 3, 1, 2)

        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class _MBConv(nn.Module):
    """Botleneck convolution layer."""

    def __init__(
        self, in_chans, out_chans, expand_ratio, activation, drop_path
    ):
        super().__init__()
        self.in_chans = in_chans
        self.hidden_chans = int(in_chans * expand_ratio)
        self.out_chans = out_chans

        self.conv1 = ConvBN2d(in_chans, self.hidden_chans, kernel_size=1)
        self.act1 = activation()
        self.conv2 = ConvBN2d(
            self.hidden_chans,
            self.hidden_chans,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=self.hidden_chans,
        )
        self.act2 = activation()
        self.conv3 = ConvBN2d(
            self.hidden_chans, out_chans, kernel_size=1, bn_weight_init=0.0
        )
        self.act3 = activation()
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        shortcut = x
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.drop_path(x)
        x += shortcut
        x = self.act3(x)
        return x


class _ConvLayer(nn.Module):
    """Convolutional layer."""

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        activation,
        drop_path=0.0,
        downsample=None,
        use_checkpoint=False,
        out_dim=None,
        conv_expand_ratio=4.0,
    ) -> None:
        """Initialize the convolutional layer.

        Args:
            dim (int): Number of input channels.
            input_resolution (tuple[int, int]): Input resolution.
            depth (int): Number of convolutional layers.
            activation (nn.Module): Activation function.
            drop_path (float): Drop path rate.
            downsample (nn.Module | None): Downsample layer.
            use_checkpoint (bool): Use checkpoint or not.
            out_dim (int | None): Number of output channels. If None, use dim.
            conv_expand_ratio (float): Expand ratio of convolutional layer.
        """
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim if out_dim is not None else dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList(
            [
                _MBConv(
                    dim,
                    dim,
                    conv_expand_ratio,
                    activation,
                    drop_path[i] if isinstance(drop_path, list) else drop_path,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution,
                dim=dim,
                out_dim=out_dim,
                activation=activation,
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        for blk in self.blocks:
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class _TinyViTBlock(nn.Module):
    """TinyViT Block."""

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        num_heads: int,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        local_conv_size: int = 3,
        activation: nn.Module = nn.GELU,
    ):
        """Initialize the TinyViT block.

        Args:
            dim (int): Number of input channels.
            input_resolution (tuple[int, int]): Image resolutions.
            num_heads (int): Number of attention heads.
            window_size (int): Window size.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            drop (float, optional): Dropout rate. Default: 0.0.
            drop_path (float, optional): Stochastic depth rate. Default: 0.0.
            local_conv_size (int): the kernel size of the convolution between
                Attention and MLP. Default: 3.
            activation: the activation function. Default: nn.GELU
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        assert window_size > 0, "window_size must be greater than 0"
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        head_dim = dim // num_heads

        window_resolution = (window_size, window_size)
        self.attn = Attention2d(
            dim,
            head_dim,
            num_heads,
            attn_ratio=1,
            resolution=window_resolution,
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_activation = activation
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            activation_layer=mlp_activation,
            dropout=drop,
        )

        pad = local_conv_size // 2
        self.local_conv = ConvBN2d(
            dim,
            dim,
            kernel_size=local_conv_size,
            stride=1,
            padding=pad,
            groups=dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        height, width = self.input_resolution
        num_batches, length, num_channels = x.shape
        assert length == height * width, "input feature has wrong size"
        res_x = x
        if height == self.window_size and width == self.window_size:
            x = self.attn(x)
        else:
            x = x.view(num_batches, height, width, num_channels)
            pad_b = (
                self.window_size - height % self.window_size
            ) % self.window_size
            pad_r = (
                self.window_size - width % self.window_size
            ) % self.window_size
            padding = pad_b > 0 or pad_r > 0

            if padding:
                x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            patch_height, patch_width = height + pad_b, width + pad_r
            num_in_height = patch_height // self.window_size
            num_in_width = patch_width // self.window_size
            # window partition
            x = (
                x.view(
                    num_batches,
                    num_in_height,
                    self.window_size,
                    num_in_width,
                    self.window_size,
                    num_channels,
                )
                .transpose(2, 3)
                .reshape(
                    num_batches * num_in_height * num_in_width,
                    self.window_size * self.window_size,
                    num_channels,
                )
            )
            x = self.attn(x)
            # window reverse
            x = (
                x.view(
                    num_batches,
                    num_in_height,
                    num_in_width,
                    self.window_size,
                    self.window_size,
                    num_channels,
                )
                .transpose(2, 3)
                .reshape(num_batches, patch_height, patch_width, num_channels)
            )

            if padding:
                x = x[:, :height, :width].contiguous()
            x = x.view(num_batches, length, num_channels)

        x = res_x + self.drop_path(x)
        x = x.transpose(1, 2).reshape(num_batches, num_channels, height, width)
        x = self.local_conv(x)
        x = x.view(num_batches, num_channels, length).transpose(1, 2)
        x = x + self.drop_path(self.mlp(x))
        return x


class _BasicLayer(nn.Module):
    """A basic TinyViT layer for one stage."""

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        downsample: nn.Module | None = None,
        use_checkpoint: bool = False,
        local_conv_size: int = 3,
        activation: type[nn.Module] = nn.GELU,
        out_dim: int | None = None,
    ) -> None:
        """Init TinyViT layer.

        Args:
            dim (int): Number of input channels.
            input_resolution (tuple[int, int]): Input resolution.
            depth (int): Number of blocks.
            num_heads (int): Number of attention heads.
            window_size (int): Local window size.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            drop (float, optional): Dropout rate. Default: 0.0
            drop_path (float | tuple[float], optional): Stochastic depth rate.
                Default: 0.0
            downsample (nn.Module | None, optional): Downsample layer at the
                end of the layer. Default: None
            use_checkpoint (bool): Whether to use checkpointing to save memory.
                Default: False.
            local_conv_size: the kernel size of the depthwise convolution
                between attention and MLP. Default: 3
            activation: the activation function. Default: nn.GELU
            out_dim: the output dimension of the layer. Default: dim
        """
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim if out_dim is not None else dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList(
            [
                _TinyViTBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    local_conv_size=local_conv_size,
                    activation=activation,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution,
                dim=dim,
                out_dim=out_dim,
                activation=activation,
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        for blk in self.blocks:
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class _TinyViT(nn.Module):
    """TinyViT backbone."""

    def __init__(
        self,
        img_size: int = 224,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dims: tuple[int, ...] = (96, 192, 384, 768),
        depths: tuple[int, ...] = (2, 2, 6, 2),
        num_heads: tuple[int, ...] = (3, 6, 12, 24),
        window_sizes: tuple[int, ...] = (7, 7, 14, 7),
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        use_checkpoint: bool = False,
        mbconv_expand_ratio: float = 4.0,
        local_conv_size: int = 3,
        layer_lr_decay: float = 1.0,
    ):
        """Init TinyViT backbone.

        Args:
            img_size (int): Input image size. Default: 224.
            in_channels (int): Number of input channels. Default: 3.
            num_classes (int): Number of classes for classification.
                Default: 1000.
            embed_dims (tuple[int]): Embedding dimension. Default: (96, 192,
                384, 768).
            depths (tuple[int]): Depth of each stage. Default: (2, 2, 6, 2).
            num_heads (tuple[int]): Number of attention heads of each stage.
                Default: (3, 6, 12, 24).
            window_sizes (tuple[int]): Local window size of each stage.
                Default: (7, 7, 14, 7).
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
                Default: 4.0.
            drop_rate (float): Dropout rate. Default: 0.0.
            drop_path_rate (float): Stochastic depth rate. Default: 0.1.
            use_checkpoint (bool): Whether to use checkpointing to save
                memory. Default: False.
            mbconv_expand_ratio (float): The expand ratio of the
                MBConv's hidden dimension. Default: 4.0.
            local_conv_size: the kernel size of the depthwise convolution
                between attention and MLP. Default: 3
            layer_lr_decay (float): The layer-wise learning rate decay.
                Default: 1.0.
        """
        super().__init__()

        self.num_classes = num_classes
        self.depths = depths
        self.num_layers = len(depths)
        self.mlp_ratio = mlp_ratio
        activation = nn.GELU

        self.patch_embed = _PatchEmbed(
            in_channels=in_channels,
            embed_dim=embed_dims[0],
            resolution=(img_size, img_size),
            activation=activation,
        )

        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            kwargs = dict(
                dim=embed_dims[i_layer],
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                ),
                depth=depths[i_layer],
                drop_path=dpr[
                    sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                ],
                downsample=_PatchMerging
                if (i_layer < self.num_layers - 1)
                else None,
                use_checkpoint=use_checkpoint,
                out_dim=embed_dims[min(i_layer + 1, len(embed_dims) - 1)],
                activation=activation,
            )
            if i_layer == 0:
                layer = _ConvLayer(
                    conv_expand_ratio=mbconv_expand_ratio,
                    **kwargs,
                )
            else:
                layer = _BasicLayer(
                    num_heads=num_heads[i_layer],
                    window_size=window_sizes[i_layer],
                    mlp_ratio=self.mlp_ratio,
                    drop=drop_rate,
                    local_conv_size=local_conv_size,
                    **kwargs,
                )
            self.layers.append(layer)

        # init weights
        self.apply(self._init_weights)
        self.set_layer_lr_decay(layer_lr_decay)

    def set_layer_lr_decay(self, layer_lr_decay: float) -> None:
        """Set the layer-wise learning rate decay."""
        decay_rate = layer_lr_decay

        # layers -> blocks (depth)
        depth = sum(self.depths)
        lr_scales = [decay_rate ** (depth - i - 1) for i in range(depth)]
        # print("LR SCALES:", lr_scales)

        def _set_lr_scale(model: nn.Module, scale: float) -> None:
            for param in model.parameters():
                param.lr_scale = scale

        self.patch_embed.apply(lambda x: _set_lr_scale(x, lr_scales[0]))
        layer_i = 0
        for layer in self.layers:
            for block in layer.blocks:
                block.apply(lambda x: _set_lr_scale(x, lr_scales[layer_i]))
                layer_i += 1
            if layer.downsample is not None:
                layer.downsample.apply(
                    lambda x: _set_lr_scale(x, lr_scales[layer_i - 1])
                )
        assert layer_i == depth
        for k, param in self.named_parameters():
            param.param_name = k

        def _check_lr_scale(model: nn.Module) -> None:
            for p in model.parameters():
                assert hasattr(p, "lr_scale"), p.param_name

        self.apply(_check_lr_scale)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        """Return a set of keywords that should not be decayed."""
        return {"attention_biases"}

    def forward_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Features extractor.

        Args:
            x (torch.Tensor): Input tensor with shape (num_batches,
                num_channels, height, width).

        Returns:
            torch.Tensor: Output tensor with shape (num_batches, num_channels,
                height', width').
        """
        x = self.patch_embed(x)
        x = self.layers[0](x)
        outs = [x]
        start_i = 1
        for i in range(start_i, len(self.layers)):
            layer = self.layers[i]
            outs.append(layer(outs[-1]))
        return outs

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass."""
        outs = self.forward_features(x)
        return outs


class TinyViT(BaseModel):
    """TinyViT model."""

    def __init__(
        self,
        vit_name: str,
        num_classes: int = 1000,
        drop_path_rate: float = 0.0,
        pretrained: bool = False,
        **kwargs,
    ):
        """Initialize TinyViT model.

        Args:
            vit_name (str): Name of the variant of the ViT model.
            num_classes (int, optional): Number of classes. Defaults to 1000.
            drop_path_rate (float, optional): Drop path rate. Defaults to 0.0.
            pretrained (bool, optional): Whether to load pretrained weights.
                Defaults to False.
            kwargs: Additional arguments to pass to the model.
        """
        super().__init__()
        self.vit_name = vit_name
        self.pretrained = pretrained

        if self.vit_name == "tiny_vit_5m_224":
            self.vit = _TinyViT(
                num_classes=num_classes,
                embed_dims=(64, 128, 160, 320),
                depths=(2, 2, 6, 2),
                num_heads=(2, 4, 5, 10),
                window_sizes=(7, 7, 14, 7),
                drop_path_rate=drop_path_rate,
                **kwargs,
            )
        elif self.vit_name == "tiny_vit_11m_224":
            self.vit = _TinyViT(
                num_classes=num_classes,
                embed_dims=(64, 128, 256, 448),
                depths=(2, 2, 6, 2),
                num_heads=(2, 4, 8, 14),
                window_sizes=(7, 7, 14, 7),
                drop_path_rate=drop_path_rate,
                **kwargs,
            )
        elif self.vit_name == "tiny_vit_21m_224":
            self.vit = _TinyViT(
                num_classes=num_classes,
                embed_dims=(96, 192, 384, 576),
                depths=(2, 2, 6, 2),
                num_heads=(3, 6, 12, 18),
                window_sizes=(7, 7, 14, 7),
                drop_path_rate=drop_path_rate,
                **kwargs,
            )
        elif self.vit_name == "tiny_vit_21m_384":
            self.vit = _TinyViT(
                num_classes=num_classes,
                embed_dims=(96, 192, 384, 576),
                depths=(2, 2, 6, 2),
                num_heads=(3, 6, 12, 18),
                window_sizes=(12, 12, 24, 12),
                drop_path_rate=drop_path_rate,
                **kwargs,
            )
        elif self.vit_name == "tiny_vit_21m_512":
            self.vit = _TinyViT(
                num_classes=num_classes,
                embed_dims=(96, 192, 384, 576),
                depths=(2, 2, 6, 2),
                num_heads=(3, 6, 12, 18),
                window_sizes=(16, 16, 32, 16),
                drop_path_rate=drop_path_rate,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown TinyVit variant: {self.vit_name}.")

    @property
    def out_channels(self) -> list[int]:
        """Return number of output channels."""
        out_channels = []
        for layer in self.vit.layers:
            out_channels.append(layer.out_dim)
        return out_channels

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass."""
        return self.vit.forward(images)
