"""Base data connectors."""

from .common import CONN_IMAGES_TEST, CONN_IMAGES_TRAIN
from .detection import CONN_BBOX_2D_TEST, CONN_BBOX_2D_TRAIN, CONN_BOX_LOSS_2D
from .visualizers import (
    CONN_BBOX_2D_TRACK_VIS,
    CONN_BBOX_2D_VIS,
    CONN_INS_MASK_2D_VIS,
)

__all__ = [
    "CONN_IMAGES_TEST",
    "CONN_IMAGES_TRAIN",
    "CONN_BBOX_2D_TEST",
    "CONN_BBOX_2D_TRAIN",
    "CONN_BOX_LOSS_2D",
    "CONN_BBOX_2D_VIS",
    "CONN_BBOX_2D_TRACK_VIS",
    "CONN_INS_MASK_2D_VIS",
]
