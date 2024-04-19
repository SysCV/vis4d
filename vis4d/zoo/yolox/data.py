# pylint: disable=duplicate-code
"""COCO data loading config for YOLOX object detection."""
from __future__ import annotations

from collections.abc import Sequence

from ml_collections import ConfigDict

from vis4d.config import class_config
from vis4d.config.typing import DataConfig
from vis4d.data.const import CommonKeys as K
from vis4d.data.data_pipe import DataPipe, MultiSampleDataPipe
from vis4d.data.datasets.coco import COCO
from vis4d.data.io import DataBackend
from vis4d.data.loader import build_train_dataloader, default_collate
from vis4d.data.transforms.affine import (
    AffineBoxes2D,
    AffineImages,
    GenAffineParameters,
)
from vis4d.data.transforms.base import RandomApply, compose
from vis4d.data.transforms.flip import FlipBoxes2D, FlipImages
from vis4d.data.transforms.mixup import (
    GenMixupParameters,
    MixupBoxes2D,
    MixupImages,
)
from vis4d.data.transforms.mosaic import (
    GenMosaicParameters,
    MosaicBoxes2D,
    MosaicImages,
)
from vis4d.data.transforms.pad import PadImages
from vis4d.data.transforms.photometric import RandomHSV
from vis4d.data.transforms.post_process import PostProcessBoxes2D
from vis4d.data.transforms.resize import (
    GenResizeParameters,
    ResizeBoxes2D,
    ResizeImages,
)
from vis4d.data.transforms.to_tensor import ToTensor
from vis4d.engine.connectors import data_key, pred_key
from vis4d.zoo.base import get_inference_dataloaders_cfg
from vis4d.zoo.base.callable import get_callable_cfg

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
    use_mixup: bool,
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
        [
            class_config(GenMosaicParameters, out_shape=image_size),
            class_config(MosaicImages, imresize_backend="cv2"),
            class_config(MosaicBoxes2D),
        ]
    ]

    preprocess_transforms += [
        [
            class_config(
                GenAffineParameters,
                scaling_ratio_range=scaling_ratio_range,
                border=(-image_size[0] // 2, -image_size[1] // 2),
            ),
            class_config(AffineImages, as_int=True),
            class_config(AffineBoxes2D),
        ]
    ]

    if use_mixup:
        preprocess_transforms += [
            [
                class_config(
                    GenMixupParameters,
                    out_shape=image_size,
                    mixup_ratio_dist="const",
                    scale_range=(0.8, 1.6),
                    pad_value=114.0,
                ),
                class_config(MixupImages, imresize_backend="cv2"),
                class_config(MixupBoxes2D),
            ]
        ]

    preprocess_transforms.append(
        [class_config(PostProcessBoxes2D, min_area=1.0)]
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
            class_config(
                GenResizeParameters,
                shape=image_size,
                keep_ratio=True,
                same_on_batch=False,
            ),
            class_config(ResizeImages, imresize_backend="cv2"),
            class_config(ResizeBoxes2D),
            class_config(PadImages, value=114.0, pad2square=True),
            class_config(ToTensor),
        ],
    )

    return class_config(
        build_train_dataloader,
        dataset=class_config(
            MultiSampleDataPipe,
            datasets=train_dataset_cfg,
            preprocess_fn=preprocess_transforms,
        ),
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        batchprocess_fn=train_batchprocess_cfg,
        collate_fn=get_callable_cfg(default_collate),
        pin_memory=True,
        shuffle=True,
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
        class_config(GenResizeParameters, shape=image_size, keep_ratio=True),
        class_config(ResizeImages, imresize_backend="cv2"),
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
    use_mixup: bool = True,
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
        use_mixup=use_mixup,
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
