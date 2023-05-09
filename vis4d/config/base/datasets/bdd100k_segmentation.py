"""COCO data loading config for object detection."""
from __future__ import annotations

from collections.abc import Sequence

from ml_collections.config_dict import ConfigDict

from vis4d.config.default.dataloader import get_dataloader_config
from vis4d.config.util import class_config
from vis4d.data.const import CommonKeys as K
from vis4d.data.datasets.bdd100k import BDD100K
from vis4d.data.io import DataBackend
from vis4d.data.transforms.base import RandomApply, compose, compose_batch
from vis4d.data.transforms.crop import (
    CropImage,
    CropSegMasks,
    GenCropParameters,
)
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
from vis4d.engine.connectors import data_key, pred_key

CONN_BDD100K_SEG_EVAL = {
    "data_names": data_key("name"),
    "masks_list": pred_key("masks"),
}


def get_train_dataloader(
    data_root: str,
    anno_path: str,
    keys_to_load: Sequence[str] = (K.images, K.seg_masks),
    data_backend: None | DataBackend = None,
    image_size: tuple[int, int] = (720, 1280),
    crop_size: tuple[int, int] = (512, 1024),
    samples_per_gpu: int = 2,
    workers_per_gpu: int = 2,
) -> ConfigDict:
    """Get the default train dataloader for BDD100K segmentation."""
    # Train Dataset
    train_dataset_cfg = class_config(
        BDD100K,
        data_root=data_root,
        annotation_path=anno_path,
        config_path="sem_seg",
        keys_to_load=keys_to_load,
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

    preprocess_transforms = [
        class_config(GenCropParameters, shape=crop_size, cat_max_ratio=0.75),
        class_config(CropImage),
        class_config(CropSegMasks),
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
            class_config(PadImages, shape=crop_size),
            class_config(PadSegMasks, shape=crop_size),
            class_config(ToTensor),
        ],
    )

    return get_dataloader_config(
        preprocess_cfg=train_preprocess_cfg,
        dataset_cfg=train_dataset_cfg,
        batchprocess_cfg=train_batchprocess_cfg,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        shuffle=True,
    )


def get_test_dataloader(
    data_root: str,
    anno_path: str,
    keys_to_load: Sequence[str] = (K.images, K.seg_masks),
    data_backend: None | DataBackend = None,
    image_size: tuple[int, int] = (720, 1280),
    samples_per_gpu: int = 1,
    workers_per_gpu: int = 1,
) -> ConfigDict:
    """Get the default test dataloader for BDD100K segmentation."""
    # Test Dataset
    test_dataset_cfg = class_config(
        BDD100K,
        data_root=data_root,
        annotation_path=anno_path,
        config_path="sem_seg",
        keys_to_load=keys_to_load,
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

    return get_dataloader_config(
        preprocess_cfg=test_preprocess_cfg,
        dataset_cfg=test_dataset_cfg,
        batchprocess_cfg=test_batchprocess_cfg,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        train=False,
    )


def get_bdd100k_segmentation_config(
    data_root: str = "data/bdd100k/images/10k",
    train_split: str = "train",
    train_keys_to_load: Sequence[str] = (K.images, K.seg_masks),
    test_split: str = "val",
    test_keys_to_load: Sequence[str] = (K.images, K.seg_masks),
    data_backend: None | ConfigDict = None,
    image_size: tuple[int, int] = (720, 1280),
    crop_size: tuple[int, int] = (512, 1024),
    samples_per_gpu: int = 2,
    workers_per_gpu: int = 2,
) -> ConfigDict:
    """Get the default config for BDD100K segmentation."""
    data = ConfigDict()

    data.train_dataloader = get_train_dataloader(
        data_root=f"{data_root}/{train_split}",
        anno_path=f"data/bdd100k/labels/sem_seg_{train_split}_rle.json",
        keys_to_load=train_keys_to_load,
        data_backend=data_backend,
        image_size=image_size,
        crop_size=crop_size,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
    )

    data.test_dataloader = get_test_dataloader(
        data_root=f"{data_root}/{test_split}",
        anno_path=f"data/bdd100k/labels/sem_seg_{test_split}_rle.json",
        keys_to_load=test_keys_to_load,
        data_backend=data_backend,
        image_size=image_size,
        samples_per_gpu=1,
        workers_per_gpu=workers_per_gpu,
    )

    return data
