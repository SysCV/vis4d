"""NuScenes multi-sensor video dataset config."""

from __future__ import annotations

from collections.abc import Sequence

from ml_collections import ConfigDict

from vis4d.config import class_config
from vis4d.data.const import CommonKeys as K
from vis4d.data.datasets.nuscenes import NuScenes


def get_nusc_train_cfg(
    data_root: str = "data/nuscenes",
    keys_to_load: Sequence[str] = (K.images, K.boxes2d, K.boxes3d),
    skip_empty_samples: bool = True,
    cache_as_binary: bool = True,
    cached_file_path: str | None = None,
    data_backend: None | ConfigDict = None,
) -> ConfigDict:
    """Get the nuScenes validation dataset config."""
    if cache_as_binary and cached_file_path is None:
        cached_file_path = f"{data_root}/train.pkl"

    return class_config(
        NuScenes,
        data_root=data_root,
        keys_to_load=keys_to_load,
        version="v1.0-trainval",
        split="train",
        skip_empty_samples=skip_empty_samples,
        data_backend=data_backend,
        cache_as_binary=cache_as_binary,
        cached_file_path=cached_file_path,
    )


def get_nusc_mini_train_cfg(
    data_root: str = "data/nuscenes",
    keys_to_load: Sequence[str] = (K.images, K.boxes2d, K.boxes3d),
    skip_empty_samples: bool = True,
    cache_as_binary: bool = True,
    cached_file_path: str | None = None,
    data_backend: None | ConfigDict = None,
) -> ConfigDict:
    """Get the nuScenes validation dataset config."""
    if cache_as_binary and cached_file_path is None:
        cached_file_path = f"{data_root}/mini_train.pkl"

    return class_config(
        NuScenes,
        data_root=data_root,
        keys_to_load=keys_to_load,
        version="v1.0-mini",
        split="mini_train",
        skip_empty_samples=skip_empty_samples,
        data_backend=data_backend,
        cache_as_binary=cache_as_binary,
        cached_file_path=cached_file_path,
    )


def get_nusc_val_cfg(
    data_root: str = "data/nuscenes",
    keys_to_load: Sequence[str] = (K.images, K.original_images, K.boxes3d),
    skip_empty_samples: bool = False,
    cache_as_binary: bool = True,
    cached_file_path: str | None = None,
    image_channel_mode: str = "RGB",
    data_backend: None | ConfigDict = None,
) -> ConfigDict:
    """Get the nuScenes validation dataset config."""
    if cache_as_binary and cached_file_path is None:
        cached_file_path = f"{data_root}/val.pkl"

    return class_config(
        NuScenes,
        data_root=data_root,
        image_channel_mode=image_channel_mode,
        keys_to_load=keys_to_load,
        version="v1.0-trainval",
        split="val",
        skip_empty_samples=skip_empty_samples,
        data_backend=data_backend,
        cache_as_binary=cache_as_binary,
        cached_file_path=cached_file_path,
    )


def get_nusc_mini_val_cfg(
    data_root: str = "data/nuscenes",
    keys_to_load: Sequence[str] = (K.images, K.original_images, K.boxes3d),
    skip_empty_samples: bool = False,
    cache_as_binary: bool = True,
    cached_file_path: str | None = None,
    image_channel_mode: str = "RGB",
    data_backend: None | ConfigDict = None,
) -> ConfigDict:
    """Get the nuScenes mini validation dataset config."""
    if cache_as_binary and cached_file_path is None:
        cached_file_path = f"{data_root}/mini_val.pkl"

    return class_config(
        NuScenes,
        data_root=data_root,
        image_channel_mode=image_channel_mode,
        keys_to_load=keys_to_load,
        version="v1.0-mini",
        split="mini_val",
        skip_empty_samples=skip_empty_samples,
        data_backend=data_backend,
        cache_as_binary=cache_as_binary,
        cached_file_path=cached_file_path,
    )
