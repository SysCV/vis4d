"""NuScenes dataset config."""

from .nuscenes import (
    get_nusc_mini_train_cfg,
    get_nusc_mini_val_cfg,
    get_nusc_train_cfg,
    get_nusc_val_cfg,
)
from .nuscenes_mono import (
    get_nusc_mono_mini_train_cfg,
    get_nusc_mono_train_cfg,
)

__all__ = [
    "get_nusc_train_cfg",
    "get_nusc_mini_train_cfg",
    "get_nusc_val_cfg",
    "get_nusc_mini_val_cfg",
    "get_nusc_mono_train_cfg",
    "get_nusc_mono_mini_train_cfg",
]
