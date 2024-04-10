"""COCO dataset config."""

from .detection import (
    CONN_COCO_BBOX_EVAL,
    CONN_COCO_MASK_EVAL,
    get_coco_detection_cfg,
)
from .sem_seg import get_coco_sem_seg_cfg

__all__ = [
    "get_coco_detection_cfg",
    "CONN_COCO_BBOX_EVAL",
    "CONN_COCO_MASK_EVAL",
    "get_coco_sem_seg_cfg",
]
