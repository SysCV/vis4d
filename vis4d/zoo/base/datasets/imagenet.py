"""ImageNet classification config."""

from __future__ import annotations

from collections.abc import Sequence

from ml_collections import ConfigDict

from vis4d.config import class_config
from vis4d.config.typing import DataConfig
from vis4d.data.const import CommonKeys as K
from vis4d.data.datasets.imagenet import ImageNet
from vis4d.data.transforms.autoaugment import RandAug
from vis4d.data.transforms.base import RandomApply, compose
from vis4d.data.transforms.crop import (
    CropImages,
    GenCentralCropParameters,
    GenRandomSizeCropParameters,
)
from vis4d.data.transforms.flip import FlipImages
from vis4d.data.transforms.mixup import (
    GenMixupParameters,
    MixupCategories,
    MixupImages,
)
from vis4d.data.transforms.normalize import NormalizeImages
from vis4d.data.transforms.random_erasing import RandomErasing
from vis4d.data.transforms.resize import GenResizeParameters, ResizeImages
from vis4d.data.transforms.to_tensor import ToTensor
from vis4d.engine.connectors import data_key, pred_key
from vis4d.zoo.base import get_train_dataloader_cfg

CONN_IMAGENET_CLS_EVAL = {
    "prediction": pred_key("probs"),
    "groundtruth": data_key("categories"),
}


def get_train_dataloader(
    data_root: str,
    split: str,
    keys_to_load: Sequence[str],
    image_size: tuple[int, int],
    samples_per_gpu: int,
    workers_per_gpu: int,
) -> ConfigDict:
    """Get the default train dataloader for ImageNet 1K dataset."""
    # Train Dataset
    train_dataset_cfg = class_config(
        ImageNet,
        data_root=data_root,
        split=split,
        num_classes=1000,
        keys_to_load=keys_to_load,
    )

    flip_trans = class_config(
        RandomApply,
        transforms=[class_config(FlipImages)],
        probability=0.5,
    )
    random_resized_crop_trans = [
        class_config(GenRandomSizeCropParameters),
        class_config(CropImages),
        class_config(GenResizeParameters, shape=image_size, keep_ratio=False),
        class_config(ResizeImages),
    ]
    random_aug_trans = [
        class_config(RandAug, magnitude=10, use_increasing=True),
        class_config(RandomErasing),
    ]
    normalize_trans = class_config(NormalizeImages)
    train_preprocess_cfg = class_config(
        compose,
        transforms=[
            flip_trans,
            *random_resized_crop_trans,
            *random_aug_trans,
            normalize_trans,
        ],
    )

    mixup_trans = [
        class_config(GenMixupParameters, alpha=0.2, out_shape=image_size),
        class_config(MixupImages),
        class_config(MixupCategories, num_classes=1000, label_smoothing=0.1),
    ]
    train_batchprocess_cfg = class_config(
        compose,
        transforms=[
            *mixup_trans,
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
    image_size: tuple[int, int],
    samples_per_gpu: int,
    workers_per_gpu: int,
    crop_pct: float = 0.875,
) -> ConfigDict:
    """Get the default test dataloader for COCO detection."""
    # Test Dataset

    test_dataset_cfg = class_config(
        ImageNet,
        data_root=data_root,
        split=split,
        num_classes=1000,
        keys_to_load=keys_to_load,
    )

    crop_size = tuple(int(size / crop_pct) for size in image_size)
    resized_crop_trans = [
        class_config(
            GenResizeParameters,
            shape=crop_size,
            keep_ratio=True,
            allow_overflow=True,
        ),
        class_config(ResizeImages),
        class_config(
            GenCentralCropParameters, shape=image_size, keep_ratio=False
        ),
        class_config(CropImages),
    ]
    normalize_trans = class_config(NormalizeImages)
    test_preprocess_cfg = class_config(
        compose,
        transforms=[
            *resized_crop_trans,
            normalize_trans,
        ],
    )

    mixup_trans = [
        class_config(GenMixupParameters, alpha=0.2, out_shape=image_size),
        class_config(MixupImages),
        class_config(MixupCategories, num_classes=1000, label_smoothing=0.1),
    ]
    test_batchprocess_cfg = class_config(
        compose,
        transforms=[
            *mixup_trans,
            class_config(ToTensor),
        ],
    )

    return get_train_dataloader_cfg(
        preprocess_cfg=test_preprocess_cfg,
        dataset_cfg=test_dataset_cfg,
        batchprocess_cfg=test_batchprocess_cfg,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        shuffle=False,
    )


def get_imagenet_cls_cfg(
    data_root: str = "data/imagenet",
    train_split: str = "train",
    train_keys_to_load: Sequence[str] = (
        K.images,
        K.categories,
    ),
    test_split: str = "val",
    test_keys_to_load: Sequence[str] = (
        K.images,
        K.categories,
    ),
    image_size: tuple[int, int] = (224, 224),
    samples_per_gpu: int = 256,
    workers_per_gpu: int = 8,
) -> DataConfig:
    """Get the default config for COCO detection."""
    data = DataConfig()

    data.train_dataloader = get_train_dataloader(
        data_root=data_root,
        split=train_split,
        keys_to_load=train_keys_to_load,
        image_size=image_size,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
    )

    data.test_dataloader = get_test_dataloader(
        data_root=data_root,
        split=test_split,
        keys_to_load=test_keys_to_load,
        image_size=image_size,
        samples_per_gpu=1,
        workers_per_gpu=workers_per_gpu,
    )

    return data
