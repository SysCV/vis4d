"""ViT for classification tasks."""
from __future__ import annotations

import torch
from torch import nn

from vis4d.common import ArgsType
from vis4d.engine.ckpt import load_model_checkpoint
from vis4d.model.classification.common import ClsOut
from vis4d.op.base import TorchVisionViT
from timm.models.vision_transformer import VisionTransformer as TimmViT


class ClassificationViT(nn.Module):
    """ViT for classification tasks."""

    def __init__(
        self,
        num_classes: int,
        representation_size: int | None = None,
        style: str = "timm",
        **kwargs: ArgsType,
    ) -> None:
        """Initialize the classification ViT.

        Args:
            num_classes (int): Number of classes.
            representation_size (int, optional): If set, use a linear layer
                to project the hidden representation to the specified size.
                Defaults to None, which uses the hidden representation as
                the final output.
            style (str, optional): Style of the ViT. Options are "timm" and
                "torchvision". Defaults to "timm".
            **kwargs: Keyword arguments passed to the ViT.
        """
        super().__init__()
        self.num_classes = num_classes
        if style == "torchvision":
            self.vit = TorchVisionViT(**kwargs)
        elif style == "timm":
            self.vit = TimmViT(num_classes=num_classes, **kwargs)
        # self.vit = ViT(**kwargs)
        self.hidden_dim = self.vit.out_channels[-1]

        if representation_size is not None:
            self.classifier = nn.Sequential(
                nn.Linear(self.hidden_dim, representation_size),
                nn.Tanh(),
                nn.Linear(representation_size, num_classes),
            )
            nn.init.zeros_(self.classifier[0].bias)
        else:
            self.classifier = nn.Linear(self.hidden_dim, num_classes)
            nn.init.zeros_(self.classifier.weight)
            nn.init.zeros_(self.classifier.bias)

    def forward(self, images: torch.Tensor) -> ClsOut:
        """Forward pass."""
        _, feats = self.vit(images)

        # Classifier "token" as used by standard language architectures
        feats = feats[:, 0]
        logits = self.classifier(feats)
        return ClsOut(
            logits=logits, probs=torch.softmax(logits.detach(), dim=-1)
        )


REV_KEYS = [
    (r"^norm\.", "fc_norm."),
]


class ClassificationViTMAE(nn.Module):
    """Vision Transformer with support for global average pooling."""

    def __init__(self, num_classes: int, weights: str | None = None, **kwargs):
        super().__init__()
        self.vit = TimmViT(
            num_classes=num_classes,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            global_pool="avg",
            **kwargs,
        )

        if weights is not None:
            load_model_checkpoint(self.vit, weights, rev_keys=REV_KEYS)

    def forward(self, images: torch.Tensor) -> ClsOut:
        logits = self.vit(images)
        return ClsOut(
            logits=logits, probs=torch.softmax(logits.detach(), dim=-1)
        )
