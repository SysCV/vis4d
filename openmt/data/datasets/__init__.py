"""Datasets module."""
from .bdd_video import register_bdd_video_instances
from .coco_video import register_coco_video_instances

__all__ = ["register_coco_video_instances", "register_bdd_video_instances"]
