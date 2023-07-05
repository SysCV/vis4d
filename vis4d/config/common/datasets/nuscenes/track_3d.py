"""NuScenes tracking dataset config."""
from __future__ import annotations

from ml_collections import ConfigDict

from vis4d.config import class_config
from vis4d.config.typing import DataConfig
from vis4d.config.util import (
    get_inference_dataloaders_cfg,
    get_train_dataloader_cfg,
)
from vis4d.data.const import CommonKeys as K
from vis4d.data.data_pipe import DataPipe
from vis4d.data.datasets.nuscenes import NuScenes
from vis4d.data.datasets.nuscenes_mono import NuScenesMono
from vis4d.data.loader import multi_sensor_collate
from vis4d.data.reference import MultiViewDataset, UniformViewSampler
from vis4d.data.transforms import RandomApply, compose
from vis4d.data.transforms.flip import (
    FlipBoxes2D,
    FlipBoxes3D,
    FlipImages,
    FlipIntrinsics,
)
from vis4d.data.transforms.normalize import NormalizeImages
from vis4d.data.transforms.post_process import PostProcessBoxes2D
from vis4d.data.transforms.pad import PadImages
from vis4d.data.transforms.resize import (
    GenerateResizeParameters,
    ResizeBoxes2D,
    ResizeImages,
    ResizeIntrinsics,
)
from vis4d.data.transforms.to_tensor import ToTensor


def get_nusc_mono_train(
    data_root: str, data_backend: None | ConfigDict = None
) -> ConfigDict:
    """Get the nuScenes monocular training dataset config."""
    return class_config(
        NuScenesMono,
        data_root=data_root,
        keys_to_load=[K.images, K.boxes2d, K.boxes3d],
        version="v1.0-trainval",
        split="train",
        data_backend=data_backend,
        skip_empty_samples=True,
        cache_as_binary=True,
        cached_file_path=f"{data_root}/mono_train.pkl",
    )


def get_nusc_mono_mini_train(
    data_root: str, data_backend: None | ConfigDict = None
) -> ConfigDict:
    """Get the nuScenes monocular mini training dataset config."""
    return class_config(
        NuScenesMono,
        data_root=data_root,
        keys_to_load=[K.images, K.boxes2d, K.boxes3d],
        version="v1.0-mini",
        split="mini_train",
        skip_empty_samples=True,
        data_backend=data_backend,
        cache_as_binary=True,
        cached_file_path=f"{data_root}/mono_mini_train.pkl",
    )


def get_nusc_val(
    data_root: str, data_backend: None | ConfigDict = None
) -> ConfigDict:
    """Get the nuScenes validation dataset config."""
    return class_config(
        NuScenes,
        data_root=data_root,
        keys_to_load=[K.images, K.original_images, K.boxes3d],
        version="v1.0-trainval",
        split="val",
        data_backend=data_backend,
        cache_as_binary=True,
        cached_file_path=f"{data_root}/val.pkl",
    )


def get_nusc_mini_val(
    data_root: str, data_backend: None | ConfigDict = None
) -> ConfigDict:
    """Get the nuScenes mini validation dataset config."""
    return class_config(
        NuScenes,
        data_root=data_root,
        keys_to_load=[K.images, K.original_images, K.boxes3d],
        version="v1.0-mini",
        split="mini_val",
        data_backend=data_backend,
        cache_as_binary=True,
        cached_file_path=f"{data_root}/mini_val.pkl",
    )


def get_train_dataloader(
    train_dataset: ConfigDict, samples_per_gpu: int, workers_per_gpu: int
) -> ConfigDict:
    """Get the default train dataloader for nuScenes tracking."""
    train_dataset_cfg = class_config(
        MultiViewDataset,
        dataset=train_dataset,
        sampler=class_config(UniformViewSampler, scope=2, num_ref_samples=1),
    )

    preprocess_transforms = [
        class_config(
            GenerateResizeParameters,
            shape=(900, 1600),
            keep_ratio=True,
        ),
        class_config(ResizeImages),
        class_config(ResizeBoxes2D),
    ]

    preprocess_transforms.append(
        class_config(
            RandomApply,
            transforms=[
                class_config(FlipImages),
                class_config(FlipIntrinsics),
                class_config(FlipBoxes2D),
                class_config(FlipBoxes3D),
            ],
            probability=0.5,
        )
    )

    preprocess_transforms.append(class_config(NormalizeImages))
    preprocess_transforms.append(class_config(PostProcessBoxes2D))

    train_preprocess_cfg = class_config(
        compose, transforms=preprocess_transforms
    )

    train_batchprocess_cfg = class_config(
        compose,
        transforms=[class_config(PadImages), class_config(ToTensor)],
    )

    return get_train_dataloader_cfg(
        preprocess_cfg=train_preprocess_cfg,
        dataset_cfg=train_dataset_cfg,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        batchprocess_cfg=train_batchprocess_cfg,
    )


def get_test_dataloader(test_dataset: ConfigDict) -> ConfigDict:
    """Get the default test dataloader for nuScenes tracking."""
    test_preprocess_cfg = class_config(
        compose,
        transforms=[
            class_config(
                GenerateResizeParameters,
                shape=(900, 1600),
                keep_ratio=True,
                sensors=NuScenes.CAMERAS,
            ),
            class_config(ResizeImages, sensors=NuScenes.CAMERAS),
            class_config(ResizeIntrinsics, sensors=NuScenes.CAMERAS),
            class_config(NormalizeImages, sensors=NuScenes.CAMERAS),
        ],
    )

    test_batchprocess_cfg = class_config(
        compose,
        transforms=[
            class_config(PadImages, sensors=NuScenes.CAMERAS),
            # class_config(NormalizeImages, sensors=NuScenes.CAMERAS),
            class_config(ToTensor, sensors=NuScenes.CAMERAS),
        ],
    )

    test_dataset_cfg = class_config(
        DataPipe, datasets=test_dataset, preprocess_fn=test_preprocess_cfg
    )

    return get_inference_dataloaders_cfg(
        datasets_cfg=test_dataset_cfg,
        video_based_inference=True,
        batchprocess_cfg=test_batchprocess_cfg,
        collate_fn=multi_sensor_collate,
    )


def get_nusc_track_cfg(
    data_root: str = "data/nuscenes",
    version: str = "v1.0-trainval",
    train_split: str = "train",
    test_split: str = "val",
    data_backend: None | ConfigDict = None,
    samples_per_gpu: int = 2,
    workers_per_gpu: int = 2,
) -> DataConfig:
    """Get the default config for nuScenes tracking."""
    data = DataConfig()

    if version == "v1.0-mini":
        assert train_split == "mini_train"
        assert test_split == "mini_val"
        train_dataset = get_nusc_mono_mini_train(
            data_root=data_root, data_backend=data_backend
        )
        test_dataset = get_nusc_mini_val(
            data_root=data_root, data_backend=data_backend
        )
    elif version == "v1.0-trainval":
        assert train_split == "train"
        assert test_split == "val"
        train_dataset = get_nusc_mono_train(
            data_root=data_root, data_backend=data_backend
        )
        test_dataset = get_nusc_val(
            data_root=data_root, data_backend=data_backend
        )
    else:
        # TODO: Add support for v1.0-test
        raise ValueError(f"Unknown version {version}")

    data.train_dataloader = get_train_dataloader(
        train_dataset=train_dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
    )

    data.test_dataloader = get_test_dataloader(test_dataset)

    return data
