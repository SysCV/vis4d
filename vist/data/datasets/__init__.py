"""Datasets module."""
from .base import BaseDatasetConfig, BaseDatasetLoader, register_dataset
from .bdd100k import BDD100K
from .coco import COCO
from .custom import Custom
from .kitti import KITTI
from .motchallenge import MOTChallenge
from .scalabel import Scalabel
from .waymo import Waymo

__all__ = [
    "register_dataset",
    "BaseDatasetLoader",
    "BaseDatasetConfig",
    "BDD100K",
    "Scalabel",
    "COCO",
    "Custom",
    "MOTChallenge",
    "Waymo",
    "KITTI",
]
