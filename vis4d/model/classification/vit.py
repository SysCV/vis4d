import vis4d.op.base import ViT
from torch import nn
import torch


class ClassificationViT(nn.Module):
    """ViT for classification tasks."""

    def __init__(self, num_classes: int, *args, **kwargs) -> None:
        """Initialize the classification ViT.

        Args:
            num_classes (int): Number of classes.
            *args: Arguments passed to the ViT.
            **kwargs: Keyword arguments passed to the ViT.
        """
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.vit = ViT(*args, **kwargs)
        self.classifier = nn.Linear(self.vit.out_channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        _, feats = self.vit(x)
        y = self.classifier(feats)
        return y
