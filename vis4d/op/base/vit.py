"""Residual networks for classification."""

from __future__ import annotations

import torch
from timm.models import named_apply
from torch import nn

from ..layer import PatchEmbed, TransformerBlock
from .base import BaseModel


def _init_weights_vit_timm(  # pylint: disable=unused-argument
    module: nn.Module, name: str
) -> None:
    """Weight initialization, original timm impl (for reproducibility)."""
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, "init_weights"):
        module.init_weights()  # type: ignore


ViT_PRESET = {  # pylint: disable=consider-using-namedtuple-or-dataclass
    "vit_tiny_patch16_224": {
        "patch_size": 16,
        "embed_dim": 192,
        "depth": 12,
        "num_heads": 3,
    },
    "vit_small_patch16_224": {
        "patch_size": 16,
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
    },
    "vit_base_patch16_224": {
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
    },
    "vit_large_patch16_224": {
        "patch_size": 16,
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
    },
    "vit_huge_patch16_224": {
        "patch_size": 16,
        "embed_dim": 1280,
        "depth": 32,
        "num_heads": 16,
    },
    "vit_small_patch32_224": {
        "patch_size": 32,
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
    },
    "vit_base_patch32_224": {
        "patch_size": 32,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
    },
    "vit_large_patch32_224": {
        "patch_size": 32,
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
    },
    "vit_huge_patch32_224": {
        "patch_size": 32,
        "embed_dim": 1280,
        "depth": 32,
        "num_heads": 16,
    },
}


class VisionTransformer(BaseModel):
    """Vision Transformer (ViT) model without classification head.

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for
        Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Adapted from:
        - pytorch vision transformer impl
        - timm vision transformer impl
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        init_values: float | None = None,
        class_token: bool = True,
        no_embed_class: bool = False,
        pre_norm: bool = False,
        pos_drop_rate: float = 0.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: nn.Module | None = None,
        act_layer: nn.Module = nn.GELU(),
    ) -> None:
        """Init VisionTransformer.

        Args:
            img_size (int, optional): Input image size. Defaults to 224.
            patch_size (int, optional): Patch size. Defaults to 16.
            in_channels (int, optional): Number of input channels. Defaults to
                3.
            num_classes (int, optional): Number of classes. Defaults to 1000.
            embed_dim (int, optional): Embedding dimension. Defaults to 768.
            depth (int, optional): Depth. Defaults to 12.
            num_heads (int, optional): Number of attention heads. Defaults to
                12.
            mlp_ratio (float, optional): Ratio of MLP hidden dim to embedding
                dim. Defaults to 4.0.
            qkv_bias (bool, optional): If to add bias to qkv. Defaults to True.
            init_values (float, optional): Initial values for layer scale.
                Defaults to None.
            class_token (bool, optional): If to add a class token. Defaults to
                True.
            no_embed_class (bool, optional): If to not embed class token.
                Defaults to False.
            pre_norm (bool, optional): If to use pre-norm. Defaults to False.
            pos_drop_rate (float, optional): Postional dropout rate. Defaults
                to 0.0.
            drop_rate (float, optional): Dropout rate. Defaults to 0.0.
            attn_drop_rate (float, optional): Attention dropout rate. Defaults
                to 0.0.
            drop_path_rate (float, optional): Drop path rate. Defaults to 0.0.
            embed_layer (nn.Module, optional): Embedding layer. Defaults to
                PatchEmbed.
            norm_layer (nn.Module, optional): Normalization layer. If None,
                nn.LayerNorm is used. Defaults to None.
            act_layer (nn.Module, optional): Activation layer. Defaults to
                nn.GELU().
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.num_depth = depth
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = (
            nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        )
        embed_len = (
            num_patches
            if no_embed_class
            else num_patches + self.num_prefix_tokens
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_len, embed_dim))
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        self.norm_pre = (
            nn.LayerNorm(embed_dim, eps=1e-6) if pre_norm else nn.Identity()
        )

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        blocks = [
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                init_values=init_values,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
            for i in range(depth)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.init_weights()

    def init_weights(self) -> None:
        """Init weights using timm's implementation."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(_init_weights_vit_timm, self)

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional embeddings."""
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then
            # concat
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat(
                    (self.cls_token.expand(x.shape[0], -1, -1), x), dim=1
                )
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat(
                    (self.cls_token.expand(x.shape[0], -1, -1), x), dim=1
                )
            x = x + self.pos_embed
        return self.pos_drop(x)

    @property
    def out_channels(self) -> list[int]:
        """Return the number of output channels per feature level."""
        return [self.embed_dim] * (self.num_depth + 1)

    def __call__(self, data: torch.Tensor) -> list[torch.Tensor]:
        """Applies the ViT encoder.

        Args:
            data (tensor): Input Images into the network shape [N, C, W, H]

        """
        return self._call_impl(data)

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass.

        Args:
            images (torch.Tensor): Input images tensor of shape (B, C, H, W).

        Returns:
            feats (list[torch.Tensor]): Features of the input images extracted
                by the ViT encoder. feats[0] is the input images, and feats[1]
                is the output of the patch embedding layer. The rest of the
                elements are the outputs of each transformer block, with the
                shape (B, N, dim), where N is the number of patches, and dim
                is the embedding dimension. The final element is the output of
                the ViT encoder.
        """
        feats = [images]
        x = self.patch_embed(images)
        x = self.norm_pre(self._pos_embed(x))
        feats.append(x)
        for blk in self.blocks:
            x = blk(x)
            feats.append(x)
        return feats
