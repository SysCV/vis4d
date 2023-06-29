"""NuScenes tracking dataset config."""
from __future__ import annotations

from ml_collections import ConfigDict

from vis4d.config import class_config
from vis4d.config.typing import DataConfig
from vis4d.config.util import get_inference_dataloaders_cfg
from vis4d.data.const import CommonKeys as K
from vis4d.data.data_pipe import DataPipe
from vis4d.data.datasets.nuscenes import NuScenes
from vis4d.data.loader import multi_sensor_collate
from vis4d.data.transforms import compose
from vis4d.data.transforms.normalize import NormalizeImages
from vis4d.data.transforms.pad import PadImages
from vis4d.data.transforms.resize import (
    GenResizeParameters,
    ResizeImages,
    ResizeIntrinsics,
)
from vis4d.data.transforms.to_tensor import ToTensor


def get_test_dataloader(
    data_root: str, version: str, split: str, data_backend: None | ConfigDict
) -> ConfigDict:
    """Get the default test dataloader for nuScenes tracking."""
    test_dataset = class_config(
        NuScenes,
        data_root=data_root,
        keys_to_load=[K.images, K.original_images, K.boxes3d],
        version=version,
        split=split,
        data_backend=data_backend,
        cache_as_binary=True,
        cached_file_path="data/nuscenes/mini_val.pkl",
        # cached_file_path="data/nuscenes/val.pkl",
    )

    test_preprocess_cfg = class_config(
        compose,
        transforms=[
            class_config(
                GenResizeParameters,
                shape=(900, 1600),
                keep_ratio=True,
                sensors=NuScenes.CAMERAS,
            ),
            class_config(
                ResizeImages,
                sensors=NuScenes.CAMERAS,
            ),
            class_config(
                ResizeIntrinsics,
                sensors=NuScenes.CAMERAS,
            ),
        ],
    )

    test_batchprocess_cfg = class_config(
        compose,
        transforms=[
            class_config(
                PadImages,
                sensors=NuScenes.CAMERAS,
            ),
            class_config(
                NormalizeImages,
                sensors=NuScenes.CAMERAS,
            ),
            class_config(
                ToTensor,
                sensors=NuScenes.CAMERAS,
            ),
        ],
    )

    test_dataset_cfg = class_config(
        DataPipe,
        datasets=test_dataset,
        preprocess_fn=test_preprocess_cfg,
    )

    return get_inference_dataloaders_cfg(
        datasets_cfg=test_dataset_cfg,
        video_based_inference=True,
        batchprocess_cfg=test_batchprocess_cfg,
        collate_fn=multi_sensor_collate,
    )


def get_nusc_track_cfg(
    data_root: str = "data/nuscenes",
    version: str = "v1.0-mini",
    test_split: str = "mini_val",
    data_backend: None | ConfigDict = None,
) -> DataConfig:
    """Get the default config for nuScenes tracking."""
    data = DataConfig()

    data.test_dataloader = get_test_dataloader(
        data_root=data_root,
        version=version,
        split=test_split,
        data_backend=data_backend,
    )

    return data
