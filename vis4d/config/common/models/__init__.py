"""Common models config."""
from .faster_rcnn import get_faster_rcnn_cfg
from .mask_rcnn import get_mask_rcnn_cfg

__all__ = ["get_faster_rcnn_cfg", "get_mask_rcnn_cfg"]
