# pylint: disable=duplicate-code
"""COCO data loading config for YOLOX object detection."""
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
from vis4d.data.data_pipe import DataPipe, MosaicDataPipe
from vis4d.data.datasets.coco import COCO
from vis4d.data.io import DataBackend
from vis4d.data.transforms.affine import (
    AffineBoxes2D,
    AffineImages,
    GenAffineParameters,
)
from vis4d.data.transforms.base import RandomApply, compose
from vis4d.data.transforms.flip import FlipBoxes2D, FlipImages
from vis4d.data.transforms.mosaic import (
    GenMosaicParameters,
    MosaicBoxes2D,
    MosaicImages,
)
from vis4d.data.transforms.pad import PadImages
from vis4d.data.transforms.photometric import RandomHSV
from vis4d.data.transforms.resize import (
    GenResizeParameters,
    ResizeBoxes2D,
    ResizeImages,
)
from vis4d.data.transforms.to_tensor import ToTensor
from vis4d.engine.connectors import data_key, pred_key

CONN_COCO_BBOX_EVAL = {
    "coco_image_id": data_key(K.sample_names),
    "pred_boxes": pred_key("boxes"),
    "pred_scores": pred_key("scores"),
    "pred_classes": pred_key("class_ids"),
}

CONN_COCO_MASK_EVAL = {
    "coco_image_id": data_key(K.sample_names),
    "pred_boxes": pred_key("boxes.boxes"),
    "pred_scores": pred_key("boxes.scores"),
    "pred_classes": pred_key("boxes.class_ids"),
    "pred_masks": pred_key("masks"),
}


def get_train_dataloader(
    data_root: str,
    split: str,
    keys_to_load: Sequence[str],
    data_backend: None | DataBackend,
    image_size: tuple[int, int],
    scaling_ratio_range: tuple[float, float],
    resize_size: tuple[int, int],
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
        remove_empty=False,
        image_channel_mode="BGR",
        data_backend=data_backend,
    )

    # Train Preprocessing
    preprocess_transforms = [
        class_config(GenMosaicParameters, out_shape=image_size),
        class_config(MosaicImages),
        class_config(MosaicBoxes2D),
    ]

    preprocess_transforms += [
        class_config(
            GenAffineParameters,
            scaling_ratio_range=scaling_ratio_range,
            border=(-image_size[0] // 2, -image_size[1] // 2),
        ),
        class_config(AffineImages),
        class_config(AffineBoxes2D),
    ]

    train_preprocess_cfg = class_config(
        compose, transforms=preprocess_transforms
    )

    train_batchprocess_cfg = class_config(
        compose,
        transforms=[
            class_config(RandomHSV, same_on_batch=False),
            class_config(
                RandomApply,
                transforms=[
                    class_config(FlipImages),
                    class_config(FlipBoxes2D),
                ],
                probability=0.5,
                same_on_batch=False,
            ),
            class_config(  # ensure same size for all images in batch
                GenResizeParameters,
                shape=[
                    (resize_size[0] + i, resize_size[1] + i)
                    for i in range(0, 321, 32)
                ],
                multiscale_mode="list",
                keep_ratio=True,
            ),
            class_config(ResizeImages),
            class_config(ResizeBoxes2D),
            class_config(PadImages, value=114.0, pad2square=True),
            class_config(ToTensor),
        ],
    )

    return get_train_dataloader_cfg(
        preprocess_cfg=train_preprocess_cfg,
        dataset_cfg=train_dataset_cfg,
        data_pipe=MosaicDataPipe,
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
        image_channel_mode="BGR",
        data_backend=data_backend,
    )

    # Test Preprocessing
    preprocess_transforms = [
        class_config(
            GenResizeParameters,
            shape=image_size,
            keep_ratio=True,
            align_long_edge=True,
        ),
        class_config(ResizeImages),
    ]

    test_preprocess_cfg = class_config(
        compose, transforms=preprocess_transforms
    )

    test_batchprocess_cfg = class_config(
        compose,
        transforms=[
            class_config(PadImages, value=114.0, pad2square=True),
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


def get_coco_yolox_cfg(
    data_root: str = "data/coco",
    train_split: str = "train2017",
    train_keys_to_load: Sequence[str] = (
        K.images,
        K.boxes2d,
        K.boxes2d_classes,
    ),
    test_split: str = "val2017",
    test_keys_to_load: Sequence[str] = (K.images, K.original_images),
    data_backend: None | ConfigDict = None,
    train_image_size: tuple[int, int] = (640, 640),
    scaling_ratio_range: tuple[float, float] = (0.1, 2.0),
    resize_size: tuple[int, int] = (480, 480),
    test_image_size: tuple[int, int] = (640, 640),
    samples_per_gpu: int = 2,
    workers_per_gpu: int = 2,
) -> DataConfig:
    """Get the default config for COCO detection."""
    data = DataConfig()

    data.train_dataloader = get_train_dataloader(
        data_root=data_root,
        split=train_split,
        keys_to_load=train_keys_to_load,
        data_backend=data_backend,
        image_size=train_image_size,
        scaling_ratio_range=scaling_ratio_range,
        resize_size=resize_size,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
    )

    data.test_dataloader = get_test_dataloader(
        data_root=data_root,
        split=test_split,
        keys_to_load=test_keys_to_load,
        data_backend=data_backend,
        image_size=test_image_size,
        samples_per_gpu=1,
        workers_per_gpu=workers_per_gpu,
    )

    return data
