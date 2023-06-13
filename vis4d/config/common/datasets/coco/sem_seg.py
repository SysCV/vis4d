# pylint: disable=duplicate-code
"""COCO data loading config for for semantic segmentation."""
from __future__ import annotations

from collections.abc import Sequence

from ml_collections import ConfigDict

from vis4d.config import class_config
from vis4d.config.util import (
    get_inference_dataloaders_cfg,
    get_train_dataloader_cfg,
)
from vis4d.data.const import CommonKeys as K
from vis4d.data.datasets.coco import COCO
from vis4d.data.io import DataBackend
from vis4d.data.loader import DataPipe
from vis4d.data.transforms.base import RandomApply, compose, compose_batch
from vis4d.data.transforms.flip import FlipImage, FlipSegMasks
from vis4d.data.transforms.normalize import NormalizeImage
from vis4d.data.transforms.pad import PadImages, PadSegMasks
from vis4d.data.transforms.photometric import ColorJitter
from vis4d.data.transforms.resize import (
    GenerateResizeParameters,
    ResizeImage,
    ResizeSegMasks,
)
from vis4d.data.transforms.to_tensor import ToTensor


def get_train_dataloader(
    data_root: str,
    split: str,
    keys_to_load: Sequence[str],
    data_backend: None | DataBackend,
    image_size: tuple[int, int],
    samples_per_gpu: int,
    workers_per_gpu: int,
) -> ConfigDict:
    """Get the default train dataloader for COCO detection."""
    # Train Dataset
    train_dataset_cfg = class_config(
        COCO,
        keys_to_load=keys_to_load,
        data_root=data_root,
        split=split,
        remove_empty=True,
        data_backend=data_backend,
    )

    # Train Preprocessing
    preprocess_transforms = [
        class_config(
            GenerateResizeParameters,
            shape=image_size,
            keep_ratio=True,
            scale_range=(0.5, 2.0),
        ),
        class_config(ResizeImage),
        class_config(ResizeSegMasks),
    ]

    preprocess_transforms.append(
        class_config(
            RandomApply,
            transforms=[class_config(FlipImage), class_config(FlipSegMasks)],
            probability=0.5,
        )
    )

    preprocess_transforms.append(
        class_config(
            RandomApply,
            transforms=[class_config(ColorJitter)],
            probability=0.5,
        )
    )

    preprocess_transforms.append(class_config(NormalizeImage))

    train_preprocess_cfg = class_config(
        compose, transforms=preprocess_transforms
    )

    train_batchprocess_cfg = class_config(
        compose_batch,
        transforms=[
            class_config(PadImages),
            class_config(PadSegMasks),
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
    image_size: tuple[int, int],
    samples_per_gpu: int,
    workers_per_gpu: int,
) -> ConfigDict:
    """Get the default test dataloader for COCO detection."""
    # Test Dataset
    test_dataset = class_config(
        COCO,
        keys_to_load=keys_to_load,
        data_root=data_root,
        split=split,
        data_backend=data_backend,
    )

    # Test Preprocessing
    preprocess_transforms = [
        class_config(
            GenerateResizeParameters, shape=image_size, keep_ratio=True
        ),
        class_config(ResizeImage),
        class_config(ResizeSegMasks),
    ]

    preprocess_transforms.append(class_config(NormalizeImage))

    test_preprocess_cfg = class_config(
        compose, transforms=preprocess_transforms
    )

    test_batchprocess_cfg = class_config(
        compose_batch,
        transforms=[
            class_config(PadImages, shape=image_size),
            class_config(PadSegMasks, shape=image_size),
            class_config(ToTensor),
        ],
    )

    # Test Dataset Config
    test_dataset_cfg = class_config(
        DataPipe, datasets=test_dataset, preprocess_fn=test_preprocess_cfg
    )

    return get_inference_dataloaders_cfg(
        datasets_cfg=test_dataset_cfg,
        batchprocess_cfg=test_batchprocess_cfg,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
    )


def get_coco_sem_seg_cfg(
    data_root: str = "data/coco",
    train_split: str = "train2017",
    train_keys_to_load: Sequence[str] = (K.images, K.seg_masks),
    test_split: str = "val2017",
    test_keys_to_load: Sequence[str] = (K.images, K.seg_masks),
    data_backend: None | ConfigDict = None,
    image_size: tuple[int, int] = (520, 520),
    samples_per_gpu: int = 2,
    workers_per_gpu: int = 2,
) -> ConfigDict:
    """Get the default config for COCO semantic segmentation."""
    data = ConfigDict()

    data.train_dataloader = get_train_dataloader(
        data_root=data_root,
        split=train_split,
        keys_to_load=train_keys_to_load,
        data_backend=data_backend,
        image_size=image_size,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
    )

    data.test_dataloader = get_test_dataloader(
        data_root=data_root,
        split=test_split,
        keys_to_load=test_keys_to_load,
        data_backend=data_backend,
        image_size=image_size,
        samples_per_gpu=1,
        workers_per_gpu=workers_per_gpu,
    )

    return data
