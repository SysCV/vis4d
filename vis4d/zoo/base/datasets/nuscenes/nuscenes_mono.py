"""NuScenes monocular dataset config."""

from __future__ import annotations

from collections.abc import Sequence

from ml_collections import ConfigDict

from vis4d.config import class_config
from vis4d.data.const import CommonKeys as K
from vis4d.data.datasets.nuscenes_mono import NuScenesMono


def get_nusc_mono_train_cfg(
    data_root: str = "data/nuscenes",
    keys_to_load: Sequence[str] = (K.images, K.boxes2d, K.boxes3d),
    skip_empty_samples: bool = True,
    cache_as_binary: bool = True,
    cached_file_path: str | None = None,
    data_backend: None | ConfigDict = None,
) -> ConfigDict:
    """Get the nuScenes monocular training dataset config."""
    if cache_as_binary and cached_file_path is None:
        cached_file_path = f"{data_root}/mono_train.pkl"

    return class_config(
        NuScenesMono,
        data_root=data_root,
        keys_to_load=keys_to_load,
        version="v1.0-trainval",
        split="train",
        skip_empty_samples=skip_empty_samples,
        cache_as_binary=cache_as_binary,
        cached_file_path=cached_file_path,
        data_backend=data_backend,
    )


def get_nusc_mono_mini_train_cfg(
    data_root: str = "data/nuscenes",
    keys_to_load: Sequence[str] = (K.images, K.boxes2d, K.boxes3d),
    skip_empty_samples: bool = True,
    cache_as_binary: bool = True,
    cached_file_path: str | None = None,
    data_backend: None | ConfigDict = None,
) -> ConfigDict:
    """Get the nuScenes monocular mini training dataset config."""
    if cache_as_binary and cached_file_path is None:
        cached_file_path = f"{data_root}/mono_mini_train.pkl"

    return class_config(
        NuScenesMono,
        data_root=data_root,
        keys_to_load=keys_to_load,
        version="v1.0-mini",
        split="mini_train",
        skip_empty_samples=skip_empty_samples,
        cache_as_binary=cache_as_binary,
        cached_file_path=cached_file_path,
        data_backend=data_backend,
    )
