"""COCO data loading config for object detection."""
from __future__ import annotations

from ml_collections.config_dict import ConfigDict

from vis4d.config.dataloader import get_dataloader_config
from vis4d.config.util import class_config
from vis4d.data.const import CommonKeys as K
from vis4d.data.datasets.coco import COCO
from vis4d.data.io import DataBackend
from vis4d.data.transforms.base import RandomApply, compose, compose_batch
from vis4d.data.transforms.flip import FlipBoxes2D, FlipImage
from vis4d.data.transforms.normalize import NormalizeImage
from vis4d.data.transforms.pad import PadImages
from vis4d.data.transforms.resize import (
    GenerateResizeParameters,
    ResizeBoxes2D,
    ResizeImage,
)
from vis4d.data.transforms.to_tensor import ToTensor
from vis4d.engine.connectors import data_key, pred_key

CONN_COCO_BBOX_EVAL = {
    "coco_image_id": data_key("coco_image_id"),
    "pred_boxes": pred_key("boxes"),
    "pred_scores": pred_key("scores"),
    "pred_classes": pred_key("class_ids"),
}

CONN_BBOX_2D_VIS = {
    K.images: data_key(K.images),
    "boxes": pred_key("boxes"),
}


def get_train_dataloader(
    data_root: str,
    split: str,
    data_backend: None | DataBackend,
    image_size: tuple[int, int],
    samples_per_gpu: int,
    workers_per_gpu: int,
) -> ConfigDict:
    """Get the default train dataloader for COCO detection."""
    # Train Dataset
    train_dataset_cfg = class_config(
        COCO,
        keys_to_load=(K.images, K.boxes2d, K.boxes2d_classes),
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
            align_long_edge=True,
        ),
        class_config(ResizeImage),
        class_config(ResizeBoxes2D),
    ]

    preprocess_transforms.append(
        class_config(
            RandomApply,
            transforms=[
                class_config(FlipImage),
                class_config(FlipBoxes2D),
            ],
            probability=0.5,
        )
    )

    preprocess_transforms.append(class_config(NormalizeImage))

    train_preprocess_cfg = class_config(
        compose,
        transforms=preprocess_transforms,
    )

    train_batchprocess_cfg = class_config(
        compose_batch,
        transforms=[
            class_config(PadImages),
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
    split: str,
    data_backend: None | DataBackend,
    image_size: tuple[int, int],
    samples_per_gpu: int,
    workers_per_gpu: int,
) -> ConfigDict:
    """Get the default test dataloader for COCO detection."""
    # Test Dataset
    test_dataset_cfg = class_config(
        COCO,
        keys_to_load=(K.images, K.boxes2d, K.boxes2d_classes),
        data_root=data_root,
        split=split,
        data_backend=data_backend,
    )

    # Test Preprocessing
    preprocess_transforms = [
        class_config(
            GenerateResizeParameters,
            shape=image_size,
            keep_ratio=True,
            align_long_edge=True,
        ),
        class_config(ResizeImage),
        class_config(ResizeBoxes2D),
    ]

    preprocess_transforms.append(class_config(NormalizeImage))

    test_preprocess_cfg = class_config(
        compose,
        transforms=preprocess_transforms,
    )

    test_batchprocess_cfg = class_config(
        compose_batch,
        transforms=[
            class_config(PadImages),
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


def get_coco_detection_config(
    data_root: str = "data/coco",
    train_split: str = "train2017",
    test_split: str = "val2017",
    data_backend: None | ConfigDict = None,
    image_size: tuple[int, int] = (800, 1333),
    samples_per_gpu: int = 2,
    workers_per_gpu: int = 2,
) -> ConfigDict:
    """Get the default config for COCO detection."""
    data = ConfigDict()

    data.train_dataloader = get_train_dataloader(
        data_root=data_root,
        split=train_split,
        data_backend=data_backend,
        image_size=image_size,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
    )

    data.test_dataloader = get_test_dataloader(
        data_root=data_root,
        split=test_split,
        data_backend=data_backend,
        image_size=image_size,
        samples_per_gpu=1,
        workers_per_gpu=workers_per_gpu,
    )

    return data
