"""Datasets module."""
from .base import Dataset, VideoDataset
from .coco import COCO

__all__ = [
    "Dataset",
    "VideoDataset",
    "COCO",
]
