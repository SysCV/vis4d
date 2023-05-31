"""ViT for classification tasks."""
from __future__ import annotations

import timm.models.vision_transformer as _vision_transformer
import torch
from torch import nn

from vis4d.common import ArgsType
from vis4d.model.classify.common import ClsOut
from vis4d.op.base.vit import VisionTransformer


class ViTClassifer(nn.Module):
    """ViT for classification tasks."""

    def __init__(
        self,
        variant: str = "",
        num_classes: int = 1000,
        use_global_pooling: bool = False,
        pretrained: bool = False,
        num_prefix_tokens: int = 1,
        **kwargs: ArgsType,
    ) -> None:
        """Initialize the classification ViT.

        Args:
            variant (str): Name of the ViT variant. Defaults to "". If set,
                the specified ViT will be loaded from timm or torchvision.
                For timm, the variant name should be in the format of
                "vit_{type}_{patch}_{image_size}". For torchvision, the variant
                name should be in the format of "vit_{type}_{patch}". For
                example, "vit_small_patch16_224" for timm and "vit_s_16" for
                torchvision. For timm, you can also leave the variant name
                empty and specify the model parameters in the kwargs.
            num_classes (int, optional): Number of classes. Defaults to 1000.
            use_global_pooling (bool, optional): If to use global pooling.
                Defaults to False. If set to True, the output of the ViT will
                be averaged over the spatial dimensions. Otherwise, the first
                token will be used for classification.
            pretrained (bool, optional): Whether to load ImageNet pre-trained
                weights. Defaults to False.
            num_prefix_tokens (int, optional): Number of prefix tokens.
                Defaults to 1.
            **kwargs: Keyword arguments passed to the ViT model.
        """
        super().__init__()
        self.num_classes = num_classes
        self.use_global_pooling = use_global_pooling
        self.num_prefix_tokens = num_prefix_tokens

        if variant != "":
            assert variant in _vision_transformer.__dict__, (
                f"Unknown ViT variant: {variant}. "
                f"Available Timm ViT variants are: "
                f"{list(_vision_transformer.__dict__.keys())}"
            )
            self.vit = _vision_transformer.__dict__[variant](
                pretrained=pretrained, **kwargs
            )
        else:
            self.vit = VisionTransformer(num_classes=num_classes, **kwargs)
            embed_dim = kwargs.get("embed_dim", 768)
            self.fc_norm = (
                nn.LayerNorm(embed_dim)
                if use_global_pooling
                else nn.Identity()
            )
            self.head = (
                nn.Linear(embed_dim, num_classes)
                if num_classes > 0
                else nn.Identity()
            )

    def forward(self, images: torch.Tensor) -> ClsOut:
        """Forward pass."""
        if not hasattr(self.vit, "head"):
            feats = self.vit(images)
            x = feats[-1]
            if self.use_global_pooling:
                x = x[:, self.num_prefix_tokens :].mean(dim=1)
            else:
                x = x[:, 0]
            x = self.fc_norm(x)
            logits = self.head(x)
        else:
            logits = self.vit(images)
        return ClsOut(
            logits=logits, probs=torch.softmax(logits.detach(), dim=-1)
        )
