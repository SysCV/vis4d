"""ViT for classification tasks."""
import torch
from torch import nn

from vis4d.common import ArgsType
from vis4d.op.base import ViT


class ClassificationViT(nn.Module):
    """ViT for classification tasks."""

    def __init__(self, num_classes: int, **kwargs: ArgsType) -> None:
        """Initialize the classification ViT.

        Args:
            num_classes (int): Number of classes.
            **kwargs: Keyword arguments passed to the ViT.
        """
        super().__init__()
        self.num_classes = num_classes
        self.vit = ViT(**kwargs)
        self.classifier = nn.Linear(self.vit.out_channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        _, feats = self.vit(x)
        y = self.classifier(feats)
        return y
