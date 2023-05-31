"""Residual networks for classification."""
from __future__ import annotations

from functools import partial

import torch
from torch import nn

from .base import BaseModel
from ..layer import TransformerBlock, PatchEmbed
from timm.models.layers import trunc_normal_
from timm.models.helpers import named_apply


def _init_weights_vit_timm(module: nn.Module):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, "init_weights"):
        module.init_weights()


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
        in_chans: int = 3,
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
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        embed_layer: nn.Module = PatchEmbed,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
        act_layer: nn.Module = nn.GELU,
        block_fn: nn.Module = TransformerBlock,
    ):
        """Init VisionTransformer.

        Args:
            img_size (int, optional): Input image size. Defaults to 224.
            patch_size (int, optional): Patch size. Defaults to 16.
            in_chans (int, optional): Number of input channels. Defaults to 3.
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
            drop_rate (float, optional): Dropout rate. Defaults to 0.0.
            attn_drop_rate (float, optional): Attention dropout rate. Defaults
                to 0.0.
            drop_path_rate (float, optional): Drop path rate. Defaults to 0.0.
            embed_layer (nn.Module, optional): Embedding layer. Defaults to
                PatchEmbed.
            norm_layer (nn.Module, optional): Norm layer. Defaults to None.
            act_layer (nn.Module, optional): Activation layer. Defaults to
                None.
            block_fn (nn.Module, optional): Block function. Defaults to
                TransformerBlock.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
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
        self.pos_embed = nn.Parameter(
            torch.randn(1, embed_len, embed_dim) * 0.02
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            *[
                block_fn(
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
        )

    def init_weights(self):
        """Init weights using timm's implementation."""
        trunc_normal_(self.pos_embed, std=0.02)
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

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass.

        Args:
            images (torch.Tensor): Input images tensor of shape (B, C, H, W).

        Returns:
            feats (list[torch.Tensor]): List of output feature tensors from
                each transformer block. All tensors are of shape (B, N, C'),
                where N is the number of patches and C' is the embedding
                dimension. The number of patches N is determined by the image
                size and patch size.
        """
        feats = []
        x = self.patch_embed(images)
        x = self._pos_embed(x)
        x = self.norm_pre(x)
        for blk in self.blocks:
            x = blk(x)
            feats.append(x)
        return feats
