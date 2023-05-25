"""Base dataset config."""
from .coco_detection import CONN_COCO_BBOX_EVAL, get_coco_detection_cfg

__all__ = ["get_coco_detection_cfg", "CONN_COCO_BBOX_EVAL"]
