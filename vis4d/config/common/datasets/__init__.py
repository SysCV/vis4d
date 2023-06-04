"""Base dataset config."""
from .bdd100k_track import get_bdd100k_track_cfg
from .coco_detection import CONN_COCO_BBOX_EVAL, get_coco_detection_cfg

__all__ = [
    "get_coco_detection_cfg",
    "CONN_COCO_BBOX_EVAL",
    "get_bdd100k_track_cfg",
]
