"""BDD100K tracking dataset configs."""

from __future__ import annotations

from ml_collections import ConfigDict

from vis4d.config import class_config
from vis4d.config.typing import DataConfig
from vis4d.data.const import CommonKeys as K
from vis4d.data.data_pipe import DataPipe
from vis4d.data.datasets import BDD100K, bdd100k_track_map
from vis4d.data.reference import MultiViewDataset, UniformViewSampler
from vis4d.data.transforms import RandomApply, compose
from vis4d.data.transforms.flip import FlipBoxes2D, FlipImages
from vis4d.data.transforms.normalize import NormalizeImages
from vis4d.data.transforms.pad import PadImages
from vis4d.data.transforms.post_process import PostProcessBoxes2D
from vis4d.data.transforms.resize import (
    GenResizeParameters,
    ResizeBoxes2D,
    ResizeImages,
)
from vis4d.data.transforms.to_tensor import ToTensor
from vis4d.engine.connectors import data_key, pred_key
from vis4d.zoo.base import (
    get_inference_dataloaders_cfg,
    get_train_dataloader_cfg,
)

CONN_BDD100K_TRACK_EVAL = {
    "frame_ids": data_key("frame_ids"),
    "sample_names": data_key(K.sample_names),
    "sequence_names": data_key(K.sequence_names),
    "pred_boxes": pred_key("boxes"),
    "pred_classes": pred_key("class_ids"),
    "pred_scores": pred_key("scores"),
    "pred_track_ids": pred_key("track_ids"),
}


def get_train_dataloader(
    data_backend: None | ConfigDict,
    samples_per_gpu: int,
    workers_per_gpu: int,
) -> ConfigDict:
    """Get the default train dataloader for BDD100K tracking."""
    bdd100k_det_train = class_config(
        BDD100K,
        data_root="data/bdd100k/images/100k/train/",
        keys_to_load=(K.images, K.boxes2d),
        annotation_path="data/bdd100k/labels/det_20/det_train.json",
        config_path="det",
        data_backend=data_backend,
        category_map=bdd100k_track_map,
        skip_empty_samples=True,
        cache_as_binary=True,
        cached_file_path="data/bdd100k/det_train.pkl",
    )

    bdd100k_track_train = class_config(
        BDD100K,
        data_root="data/bdd100k/images/track/train/",
        keys_to_load=(K.images, K.boxes2d),
        annotation_path="data/bdd100k/labels/box_track_20/train/",
        config_path="box_track",
        data_backend=data_backend,
        category_map=bdd100k_track_map,
        skip_empty_samples=True,
        cache_as_binary=True,
        cached_file_path="data/bdd100k/track_train.pkl",
    )

    train_dataset_cfg = [
        class_config(
            MultiViewDataset,
            dataset=bdd100k_det_train,
            sampler=class_config(
                UniformViewSampler, scope=0, num_ref_samples=1
            ),
        ),
        class_config(
            MultiViewDataset,
            dataset=bdd100k_track_train,
            sampler=class_config(
                UniformViewSampler, scope=3, num_ref_samples=1
            ),
        ),
    ]

    preprocess_transforms = [
        class_config(
            GenResizeParameters,
            shape=(720, 1280),
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
                class_config(FlipBoxes2D),
            ],
            probability=0.5,
        )
    )

    preprocess_transforms.append(class_config(NormalizeImages))
    preprocess_transforms.append(class_config(PostProcessBoxes2D))

    train_preprocess_cfg = class_config(
        compose,
        transforms=preprocess_transforms,
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


def get_test_dataloader(
    data_backend: None | ConfigDict,
    samples_per_gpu: int,
    workers_per_gpu: int,
) -> ConfigDict:
    """Get the default test dataloader for BDD100K tracking."""
    test_dataset = class_config(
        BDD100K,
        data_root="data/bdd100k/images/track/val/",
        keys_to_load=(K.images, K.original_images),
        annotation_path="data/bdd100k/labels/box_track_20/val/",
        config_path="box_track",
        category_map=bdd100k_track_map,
        data_backend=data_backend,
        cache_as_binary=True,
        cached_file_path="data/bdd100k/track_val.pkl",
    )

    preprocess_transforms = [
        class_config(
            GenResizeParameters,
            shape=(720, 1280),
            keep_ratio=True,
        ),
        class_config(ResizeImages),
        class_config(NormalizeImages),
    ]

    test_preprocess_cfg = class_config(
        compose,
        transforms=preprocess_transforms,
    )

    test_batchprocess_cfg = class_config(
        compose,
        transforms=[
            class_config(PadImages),
            class_config(ToTensor),
        ],
    )

    test_dataset_cfg = class_config(
        DataPipe,
        datasets=test_dataset,
        preprocess_fn=test_preprocess_cfg,
    )

    return get_inference_dataloaders_cfg(
        datasets_cfg=test_dataset_cfg,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        video_based_inference=True,
        batchprocess_cfg=test_batchprocess_cfg,
    )


def get_bdd100k_track_cfg(
    data_backend: None | ConfigDict = None,
    samples_per_gpu: int = 2,
    workers_per_gpu: int = 2,
) -> DataConfig:
    """Get the default config for BDD100K tracking."""
    data = DataConfig()

    data.train_dataloader = get_train_dataloader(
        data_backend=data_backend,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
    )

    data.test_dataloader = get_test_dataloader(
        data_backend=data_backend,
        samples_per_gpu=1,
        workers_per_gpu=1,
    )

    return data
