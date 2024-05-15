"""SHIFT dataset config."""

from .tasks import (
    CONN_SHIFT_DET_EVAL,
    CONN_SHIFT_INS_EVAL,
    get_shift_depth_est_config,
    get_shift_det_config,
    get_shift_instance_seg_config,
    get_shift_multitask_2d_config,
    get_shift_multitask_3d_config,
    get_shift_optical_flow_config,
    get_shift_sem_seg_config,
    get_shift_tracking_config,
)

__all__ = [
    "CONN_SHIFT_DET_EVAL",
    "CONN_SHIFT_INS_EVAL",
    "get_shift_depth_est_config",
    "get_shift_det_config",
    "get_shift_instance_seg_config",
    "get_shift_tracking_config",
    "get_shift_multitask_2d_config",
    "get_shift_multitask_3d_config",
    "get_shift_optical_flow_config",
    "get_shift_sem_seg_config",
]
