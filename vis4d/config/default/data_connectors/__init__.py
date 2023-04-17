"""Default data connection setups."""
from .common import CONN_IMAGES_TEST, CONN_IMAGES_TRAIN
from .detection import CONN_BBOX_2D_TEST, CONN_BBOX_2D_TRAIN, CONN_BOX_LOSS_2D
from .visualizers import CONN_BBOX_2D_VIS
from .static import get_static_data_connector_config

__all__ = [
    "get_static_data_connector_config",
    "CONN_IMAGES_TEST",
    "CONN_IMAGES_TRAIN",
    "CONN_BBOX_2D_TEST",
    "CONN_BBOX_2D_TRAIN",
    "CONN_BOX_LOSS_2D",
    "CONN_BBOX_2D_VIS",
]
