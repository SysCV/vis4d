"""Datasets module."""
from .base import BaseDataset, BaseVideoDataset
from .coco import COCO

__all__ = [
    "BaseDataset",
    "BaseVideoDataset",
    "COCO",
]
