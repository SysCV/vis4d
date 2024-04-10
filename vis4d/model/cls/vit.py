"""ViT for classification tasks."""

from __future__ import annotations

import timm.models.vision_transformer as _vision_transformer
import torch
from torch import nn

from vis4d.common import ArgsType
from vis4d.common.ckpt import load_model_checkpoint
from vis4d.op.base.vit import VisionTransformer, ViT_PRESET

from .common import ClsOut


class ViTClassifer(nn.Module):
    """ViT for classification tasks."""

    def __init__(
        self,
        variant: str = "",
        num_classes: int = 1000,
        use_global_pooling: bool = False,
        weights: str | None = None,
        num_prefix_tokens: int = 1,
        **kwargs: ArgsType,
    ) -> None:
        """Initialize the classification ViT.

        Args:
            variant (str): Name of the ViT variant. Defaults to "". If the name
                starts with "timm://", the variant will be loaded from timm's
                model zoo. Otherwise, the variant will be loaded from the
                ViT_PRESET dict. If the variant is empty, the default ViT
                variant will be used. In all cases, the additional keyword
                arguments will override the default arguments.
            num_classes (int, optional): Number of classes. Defaults to 1000.
            use_global_pooling (bool, optional): If to use global pooling.
                Defaults to False. If set to True, the output of the ViT will
                be averaged over the spatial dimensions. Otherwise, the first
                token will be used for classification.
            weights (str, optional): If to load pretrained weights. If set to
                "timm", the weights will be loaded from timm's model zoo that
                matches the variant. If a URL is provided, the weights will be
                downloaded from the URL. Defaults to None, which means no
                weights will be loaded.
            num_prefix_tokens (int, optional): Number of prefix tokens.
                Defaults to 1.
            **kwargs: Keyword arguments passed to the ViT model.
        """
        super().__init__()
        self.num_classes = num_classes
        self.use_global_pooling = use_global_pooling
        self.num_prefix_tokens = num_prefix_tokens

        if variant != "":
            assert variant in ViT_PRESET, (
                f"Unknown ViT variant: {variant}. "
                f"Available ViT variants are: {list(ViT_PRESET.keys())}"
            )
            preset_kwargs = ViT_PRESET[variant]
            preset_kwargs["num_classes"] = num_classes
            preset_kwargs.update(kwargs)
            self.vit = VisionTransformer(**preset_kwargs)  # type: ignore
        else:
            # Build ViT from scratch using kwargs
            preset_kwargs = {}
            self.vit = VisionTransformer(num_classes=num_classes, **kwargs)

        # Classification head
        embed_dim = kwargs.get(
            "embed_dim", preset_kwargs.get("embed_dim", 768)
        )
        self.norm = (
            nn.LayerNorm(embed_dim) if use_global_pooling else nn.Identity()
        )
        self.head = (
            nn.Linear(embed_dim, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        # Load pretrain weights
        if weights is not None:
            if weights.startswith("timm://"):
                weights = weights.removeprefix("timm://")
                if "." in weights:
                    model_name, pretrain_tag = weights.split(".")
                else:
                    model_name = weights
                    pretrain_tag = None
                assert model_name in _vision_transformer.__dict__, (
                    f"Unknown Timm ViT weights: {model_name}. "
                    f"Available Timm ViT weights are: "
                    f"{list(_vision_transformer.__dict__.keys())}"
                )
                _model = _vision_transformer.__dict__[model_name](
                    pretrained=True, pretrained_cfg=pretrain_tag, **kwargs
                )
                self.vit.load_state_dict(_model.state_dict(), strict=False)
                self.norm.load_state_dict(
                    _model.norm.state_dict(), strict=False
                )
                self.head.load_state_dict(
                    _model.head.state_dict(), strict=False
                )
            else:
                load_model_checkpoint(self, weights)

    def forward(self, images: torch.Tensor) -> ClsOut:
        """Forward pass."""
        feats = self.vit(images)
        x = feats[-1]
        if self.use_global_pooling:
            x = x[:, self.num_prefix_tokens :].mean(dim=1)
        else:
            x = x[:, 0]
        x = self.norm(x)
        logits = self.head(x)
        return ClsOut(
            logits=logits, probs=torch.softmax(logits.detach(), dim=-1)
        )
