"""Classification models for Vis4D."""

from .tinyvit import ClassificationTinyViT
from .vit import ClassificationViT

__all__ = ["ClassificationViT", "ClassificationTinyViT"]
