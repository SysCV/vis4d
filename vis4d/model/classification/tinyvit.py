"""ViT for classification tasks."""
import torch
from torch import nn

from vis4d.common import ArgsType
from vis4d.model.classification.common import ClsOut
from vis4d.op.base import TinyViT


class ClassificationTinyViT(nn.Module):
    """ViT for classification tasks."""

    def __init__(self, num_classes: int, **kwargs: ArgsType) -> None:
        """Initialize the classification ViT.

        Args:
            num_classes (int): Number of classes.
            **kwargs: Keyword arguments passed to the ViT.
        """
        super().__init__()
        self.num_classes = num_classes
        self.vit = TinyViT(num_classes=num_classes, **kwargs)
        self.hidden_dim = self.vit.out_channels[-1]
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.classifier = nn.Linear(self.hidden_dim, num_classes)

    def forward(self, images: torch.Tensor) -> ClsOut:
        """Forward pass."""
        feat = self.vit(images)[-1]
        x = feat.mean(1)
        x = self.layer_norm(x)
        logits = self.classifier(x)
        return ClsOut(logits=logits, probs=torch.softmax(logits, dim=-1))
