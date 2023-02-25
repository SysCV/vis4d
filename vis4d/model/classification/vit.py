"""ViT for classification tasks."""
from __future__ import annotations

import torch
from torch import nn

from vis4d.common import ArgsType
from vis4d.model.classification.common import ClsOut
from vis4d.op.base import ViT


class ClassificationViT(nn.Module):
    """ViT for classification tasks."""

    def __init__(
        self,
        num_classes: int,
        representation_size: int | None = None,
        **kwargs: ArgsType
    ) -> None:
        """Initialize the classification ViT.

        Args:
            num_classes (int): Number of classes.
            **kwargs: Keyword arguments passed to the ViT.
        """
        super().__init__()
        self.num_classes = num_classes
        self.vit = ViT(**kwargs)
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
        return ClsOut(logits=logits, probs=torch.softmax(logits.detach(), dim=-1))
