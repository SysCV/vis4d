"""Datasets module."""
from .coco import register_coco_instances
from .scalabel import register_scalabel_instances

__all__ = ["register_scalabel_instances", "register_coco_instances"]
