# pylint: disable=duplicate-code
"""BDD100K dataset config for object detection."""
from __future__ import annotations

from collections.abc import Sequence

from ml_collections import ConfigDict

from vis4d.config import class_config
from vis4d.config.typing import DataConfig
from vis4d.data.const import CommonKeys as K
from vis4d.data.data_pipe import DataPipe
from vis4d.data.datasets import BDD100K
from vis4d.data.io import DataBackend
from vis4d.data.transforms.base import RandomApply, compose
from vis4d.data.transforms.flip import (
    FlipBoxes2D,
    FlipImages,
    FlipInstanceMasks,
)
from vis4d.data.transforms.normalize import NormalizeImages
from vis4d.data.transforms.pad import PadImages
from vis4d.data.transforms.resize import (
    GenResizeParameters,
    ResizeBoxes2D,
    ResizeImages,
    ResizeInstanceMasks,
)
from vis4d.data.transforms.to_tensor import ToTensor
from vis4d.engine.connectors import data_key, pred_key
from vis4d.zoo.base import (
    get_inference_dataloaders_cfg,
    get_train_dataloader_cfg,
)

CONN_BDD100K_DET_EVAL = {
    "frame_ids": data_key("frame_ids"),
    "sample_names": data_key("sample_names"),
    "sequence_names": data_key("sequence_names"),
    "pred_boxes": pred_key("boxes"),
    "pred_scores": pred_key("scores"),
    "pred_classes": pred_key("class_ids"),
}
CONN_BDD100K_INS_EVAL = {
    "frame_ids": data_key("frame_ids"),
    "sample_names": data_key("sample_names"),
    "sequence_names": data_key("sequence_names"),
    "pred_boxes": pred_key("boxes.boxes"),
    "pred_scores": pred_key("boxes.scores"),
    "pred_classes": pred_key("boxes.class_ids"),
    "pred_masks": pred_key("masks.masks"),
}


def get_train_dataloader(
    data_root: str,
    anno_path: str,
    keys_to_load: Sequence[str] = (K.images, K.boxes2d),
    ins_seg: bool = False,
    data_backend: None | DataBackend = None,
    image_size: tuple[int, int] = (720, 1280),
    multi_scale: bool = False,
    samples_per_gpu: int = 2,
    workers_per_gpu: int = 2,
) -> ConfigDict:
    """Get the default train dataloader for BDD100K segmentation."""
    # Train Dataset
    train_dataset_cfg = class_config(
        BDD100K,
        data_root=data_root,
        annotation_path=anno_path,
        config_path="ins_seg" if ins_seg else "det",
        keys_to_load=keys_to_load,
        data_backend=data_backend,
        skip_empty_samples=True,
    )

    # Train Preprocessing
    if multi_scale:
        ms_shapes = [(image_size[0] - 24 * i, image_size[1]) for i in range(6)]
        preprocess_transforms = [
            class_config(
                GenResizeParameters,
                shape=ms_shapes,
                keep_ratio=True,
                multiscale_mode="list",
                align_long_edge=True,
            )
        ]
    else:
        preprocess_transforms = [
            class_config(
                GenResizeParameters,
                shape=image_size,
                keep_ratio=True,
                align_long_edge=True,
            )
        ]
    preprocess_transforms += [
        class_config(ResizeImages),
        class_config(ResizeBoxes2D),
    ]
    if K.instance_masks in keys_to_load:
        preprocess_transforms.append(class_config(ResizeInstanceMasks))

    flip_transforms = [class_config(FlipImages), class_config(FlipBoxes2D)]
    if K.instance_masks in keys_to_load:
        flip_transforms.append(class_config(FlipInstanceMasks))

    preprocess_transforms.append(
        class_config(
            RandomApply,
            transforms=flip_transforms,
            probability=0.5,
        )
    )

    preprocess_transforms.append(class_config(NormalizeImages))

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
        batchprocess_cfg=train_batchprocess_cfg,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
    )


def get_test_dataloader(
    data_root: str,
    anno_path: str,
    keys_to_load: Sequence[str] = (K.images, K.original_images),
    ins_seg: bool = False,
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
        config_path="ins_seg" if ins_seg else "det",
        keys_to_load=keys_to_load,
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

    preprocess_transforms.append(class_config(NormalizeImages))

    test_preprocess_cfg = class_config(
        compose, transforms=preprocess_transforms
    )

    test_batchprocess_cfg = class_config(
        compose,
        transforms=[class_config(PadImages), class_config(ToTensor)],
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


def get_bdd100k_detection_config(
    data_root: str = "data/bdd100k/images/100k",
    train_split: str = "train",
    train_keys_to_load: Sequence[str] = (K.images, K.boxes2d),
    test_split: str = "val",
    test_keys_to_load: Sequence[str] = (K.images, K.original_images),
    ins_seg: bool = False,
    data_backend: None | ConfigDict = None,
    image_size: tuple[int, int] = (720, 1280),
    multi_scale: bool = False,
    samples_per_gpu: int = 2,
    workers_per_gpu: int = 2,
) -> DataConfig:
    """Get the default config for BDD100K detection."""
    data = DataConfig()

    if K.instance_masks in train_keys_to_load:
        train_anno_path = "data/bdd100k/labels/ins_seg_train_rle.json"
        test_anno_path = "data/bdd100k/labels/ins_seg_val_rle.json"
    else:
        train_anno_path = "data/bdd100k/labels/det_20/det_train.json"
        test_anno_path = "data/bdd100k/labels/det_20/det_val.json"

    data.train_dataloader = get_train_dataloader(
        data_root=f"{data_root}/{train_split}",
        anno_path=train_anno_path,
        keys_to_load=train_keys_to_load,
        ins_seg=ins_seg,
        data_backend=data_backend,
        image_size=image_size,
        multi_scale=multi_scale,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
    )

    data.test_dataloader = get_test_dataloader(
        data_root=f"{data_root}/{test_split}",
        anno_path=test_anno_path,
        keys_to_load=test_keys_to_load,
        ins_seg=ins_seg,
        data_backend=data_backend,
        image_size=image_size,
        samples_per_gpu=1,
        workers_per_gpu=1,
    )

    return data
