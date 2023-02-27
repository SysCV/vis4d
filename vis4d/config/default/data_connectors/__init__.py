"""Default data connection setups."""
from .detection import (
    CONN_BBOX_2D_TEST,
    CONN_BBOX_2D_TRAIN,
    CONN_MASK_HEAD_LOSS_2D,
    CONN_ROI_LOSS_2D,
    CONN_RPN_LOSS_2D,
)
from .evaluators import CONN_COCO_BBOX_EVAL
from .visualizers import CONN_BBOX_2D_VIS

__all__ = [
    "CONN_BBOX_2D_TEST",
    "CONN_BBOX_2D_TRAIN",
    "CONN_RPN_LOSS_2D",
    "CONN_ROI_LOSS_2D",
    "CONN_COCO_BBOX_EVAL",
    "CONN_BBOX_2D_VIS",
    "CONN_MASK_HEAD_LOSS_2D",
]
