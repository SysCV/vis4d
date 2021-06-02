"""Datasets module."""
from .base import DatasetLoader, register_dataset_instances
from .bdd100k import BDD100K
from .coco import COCO
from .custom import Custom
from .motchallenge import MOTChallenge
from .scalabel import Scalabel

__all__ = [
    "register_dataset_instances",
    "DatasetLoader",
    "BDD100K",
    "Scalabel",
    "COCO",
    "Custom",
    "MOTChallenge",
]
