# pylint: disable=duplicate-code
"""S3DIS data loading config for for semantic segmentation."""
from __future__ import annotations

from collections.abc import Sequence

from ml_collections import ConfigDict

from vis4d.config import class_config
from vis4d.config.typing import DataConfig
from vis4d.config.util import (
    get_inference_dataloaders_cfg,
    get_train_dataloader_cfg,
)
from vis4d.data.const import CommonKeys as K
from vis4d.data.data_pipe import DataPipe
from vis4d.data.datasets.s3dis import S3DIS
from vis4d.data.io import DataBackend
from vis4d.data.transforms.base import compose
from vis4d.data.transforms.points import (
    ColorContrast,
    ColorDrop,
    ColorNormalize,
    GenContrastParams,
    GenScaleParams,
    PointJitter,
    PointScale,
    XYCenterZAlign,
)
from vis4d.data.transforms.to_tensor import ToTensor
from vis4d.data.transforms.voxelize import (
    GenVoxelMapping,
    VoxelizeColors,
    VoxelizePoints,
    VoxelizeSemantics,
)


def get_train_dataloader(
    data_root: str,
    split: str,
    keys_to_load: Sequence[str],
    data_backend: None | DataBackend,
    samples_per_gpu: int,
    workers_per_gpu: int,
    reps_per_epoch: int,
) -> ConfigDict:
    """Get the default train dataloader for s3dis segmentation."""
    # Train Dataset
    train_dataset_cfg = class_config(
        S3DIS,
        keys_to_load=keys_to_load,
        data_root=data_root,
        split=split,
        data_backend=data_backend,
        reps_per_epoch=reps_per_epoch,
    )

    # Train Preprocessing
    voxelization = [
        class_config(
            GenVoxelMapping,
            voxel_size=0.04,
            max_voxels=24000,
            shuffle=True,
            random_downsample=True,
        ),
        class_config(VoxelizePoints),
        class_config(VoxelizeColors),
        class_config(VoxelizeSemantics),
    ]

    color_augmentation = [
        class_config(GenContrastParams),
        class_config(ColorContrast),
        class_config(ColorDrop, proba=0.2),  # TODO, move to randomapply?
        class_config(
            ColorNormalize,
            color_mean=[0.5136457, 0.49523646, 0.44921124],
            color_std=[0.18308958, 0.18415008, 0.19252081],
        ),
    ]

    point_augmentation = [
        class_config(GenScaleParams, scale=[0.9, 1.1]),
        class_config(PointScale),
        class_config(XYCenterZAlign),
        class_config(PointJitter, sigma=0.005, clip=0.02),
    ]

    train_preprocess_cfg = class_config(
        compose,
        transforms=(*voxelization, *color_augmentation, *point_augmentation),
    )

    train_batchprocess_cfg = class_config(
        compose,
        transforms=[
            class_config(ToTensor),
        ],
    )

    return get_train_dataloader_cfg(
        preprocess_cfg=train_preprocess_cfg,
        dataset_cfg=train_dataset_cfg,
        batchprocess_cfg=train_batchprocess_cfg,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
    )


def get_test_dataloader(
    data_root: str,
    split: str,
    keys_to_load: Sequence[str],
    data_backend: None | DataBackend,
    samples_per_gpu: int,
    workers_per_gpu: int,
) -> ConfigDict:
    """Get the default test dataloader for s3dis segmentation."""
    test_dataset_cfg = class_config(
        S3DIS,
        keys_to_load=keys_to_load,
        data_root=data_root,
        split=split,
        data_backend=data_backend,
    )

    # Train Preprocessing
    voxelization = [
        class_config(
            GenVoxelMapping,
            voxel_size=0.04,
            max_voxels=None,
            shuffle=False,
            random_downsample=False,
        ),
        class_config(VoxelizePoints),
        class_config(VoxelizeColors),
        class_config(VoxelizeSemantics),
    ]

    color_augmentation = [
        class_config(
            ColorNormalize,
            color_mean=[0.5136457, 0.49523646, 0.44921124],
            color_std=[0.18308958, 0.18415008, 0.19252081],
        ),
    ]

    point_augmentation = [
        class_config(XYCenterZAlign),
    ]

    test_preprocess_cfg = class_config(
        compose,
        transforms=(*voxelization, *color_augmentation, *point_augmentation),
    )

    test_batchprocess_cfg = class_config(
        compose,
        transforms=[
            class_config(ToTensor),
        ],
    )

    # Test Dataset Config
    test_dataset_cfg = class_config(
        DataPipe, datasets=test_dataset_cfg, preprocess_fn=test_preprocess_cfg
    )

    return get_inference_dataloaders_cfg(
        datasets_cfg=test_dataset_cfg,
        batchprocess_cfg=test_batchprocess_cfg,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
    )


def get_s3dis_sem_seg_cfg(
    data_root: str = "data/s3dis",
    train_split: str = "trainNoArea5",
    train_keys_to_load: Sequence[str] = (
        K.points3d,
        K.colors3d,
        K.semantics3d,
    ),
    test_split: str = "testArea5",
    test_keys_to_load: Sequence[str] = (
        K.points3d,
        K.colors3d,
        K.semantics3d,
    ),
    data_backend: None | ConfigDict = None,
    samples_per_gpu: int = 2,
    workers_per_gpu: int = 2,
    reps_per_epoch: int = 30,
) -> DataConfig:
    """Get the default config for COCO semantic segmentation."""
    data = DataConfig()

    data.train_dataloader = get_train_dataloader(
        data_root=data_root,
        split=train_split,
        keys_to_load=train_keys_to_load,
        data_backend=data_backend,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        reps_per_epoch=reps_per_epoch,
    )

    data.test_dataloader = get_test_dataloader(
        data_root=data_root,
        split=test_split,
        keys_to_load=test_keys_to_load,
        data_backend=data_backend,
        samples_per_gpu=1,
        workers_per_gpu=workers_per_gpu,
    )

    return data
