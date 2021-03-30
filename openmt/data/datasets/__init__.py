"""Datasets module."""
from .coco_video import register_coco_video_instances
from .scalabel_video import register_scalabel_video_instances

__all__ = ["register_coco_video_instances", "register_scalabel_video_instances"]
