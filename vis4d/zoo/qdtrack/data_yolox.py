"""BDD100K data loading config for QDTrack YOLOX."""

from __future__ import annotations

from ml_collections import ConfigDict

from vis4d.config import class_config
from vis4d.config.typing import DataConfig
from vis4d.data.const import CommonKeys as K
from vis4d.data.data_pipe import DataPipe, MultiSampleDataPipe
from vis4d.data.datasets.bdd100k import BDD100K, bdd100k_track_map
from vis4d.data.loader import build_train_dataloader, default_collate
from vis4d.data.reference import MultiViewDataset, UniformViewSampler
from vis4d.data.transforms.affine import (
    AffineBoxes2D,
    AffineImages,
    GenAffineParameters,
)
from vis4d.data.transforms.base import RandomApply, compose
from vis4d.data.transforms.crop import (
    CropBoxes2D,
    CropImages,
    GenCropParameters,
)
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
from vis4d.data.transforms.normalize import NormalizeImages
from vis4d.data.transforms.pad import PadImages
from vis4d.data.transforms.photometric import RandomHSV
from vis4d.data.transforms.post_process import (
    PostProcessBoxes2D,
    RescaleTrackIDs,
)
from vis4d.data.transforms.resize import (
    GenResizeParameters,
    ResizeBoxes2D,
    ResizeImages,
)
from vis4d.data.transforms.to_tensor import ToTensor
from vis4d.zoo.base import get_inference_dataloaders_cfg
from vis4d.zoo.base.callable import get_callable_cfg


def get_train_dataloader(
    data_backend: None | ConfigDict,
    image_size: tuple[int, int],
    normalize_image: bool,
    samples_per_gpu: int,
    workers_per_gpu: int,
) -> ConfigDict:
    """Get the default train dataloader for BDD100K tracking."""
    bdd100k_det_train = class_config(
        BDD100K,
        data_root="data/bdd100k/images/100k/train/",
        keys_to_load=(K.images, K.boxes2d),
        annotation_path="data/bdd100k/labels/det_20/det_train.json",
        category_map=bdd100k_track_map,
        config_path="det",
        image_channel_mode="BGR",
        data_backend=data_backend,
        skip_empty_samples=True,
        cache_as_binary=True,
        cached_file_path="data/bdd100k/pkl/det_train.pkl",
    )

    bdd100k_track_train = class_config(
        BDD100K,
        data_root="data/bdd100k/images/track/train/",
        keys_to_load=(K.images, K.boxes2d),
        annotation_path="data/bdd100k/labels/box_track_20/train/",
        category_map=bdd100k_track_map,
        config_path="box_track",
        image_channel_mode="BGR",
        data_backend=data_backend,
        skip_empty_samples=True,
        cache_as_binary=True,
        cached_file_path="data/bdd100k/pkl/track_train.pkl",
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

    # Train Preprocessing
    preprocess_transforms = [
        [
            class_config(GenMosaicParameters, out_shape=image_size),
            class_config(MosaicImages, imresize_backend="cv2"),
            class_config(MosaicBoxes2D),
        ],
        [class_config(RescaleTrackIDs)],
    ]

    preprocess_transforms += [
        [
            class_config(
                GenAffineParameters,
                scaling_ratio_range=(0.5, 1.5),
                border=(-image_size[0] // 2, -image_size[1] // 2),
            ),
            class_config(AffineImages, as_int=True),
            class_config(AffineBoxes2D),
        ]
    ]

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
        ],
        [class_config(RescaleTrackIDs)],
    ]

    preprocess_transforms.append(
        [class_config(PostProcessBoxes2D, min_area=1.0)]
    )

    batch_transforms = [
        class_config(RandomHSV, same_on_batch=False),
        class_config(
            RandomApply,
            transforms=[class_config(FlipImages), class_config(FlipBoxes2D)],
            probability=0.5,
            same_on_batch=False,
        ),
        class_config(
            GenResizeParameters,
            shape=image_size,
            keep_ratio=True,
            scale_range=(0.5, 1.5),
            same_on_batch=False,
        ),
        class_config(ResizeImages),
        class_config(ResizeBoxes2D),
        class_config(GenCropParameters, shape=image_size, same_on_batch=False),
        class_config(CropImages),
        class_config(CropBoxes2D),
    ]
    if normalize_image:
        batch_transforms += [
            class_config(NormalizeImages),
            class_config(PadImages),
        ]
    else:
        batch_transforms += [class_config(PadImages, value=114.0)]
    train_batchprocess_cfg = class_config(
        compose, transforms=batch_transforms + [class_config(ToTensor)]
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
    data_backend: None | ConfigDict,
    image_size: tuple[int, int],
    normalize_image: bool,
    samples_per_gpu: int,
    workers_per_gpu: int,
) -> ConfigDict:
    """Get the default test dataloader for BDD100K tracking."""
    test_dataset = class_config(
        BDD100K,
        data_root="data/bdd100k/images/track/val/",
        keys_to_load=(K.images, K.original_images),
        annotation_path="data/bdd100k/labels/box_track_20/val/",
        category_map=bdd100k_track_map,
        config_path="box_track",
        image_channel_mode="BGR",
        data_backend=data_backend,
        cache_as_binary=True,
        cached_file_path="data/bdd100k/pkl/track_val.pkl",
    )

    preprocess_transforms = [
        class_config(GenResizeParameters, shape=image_size, keep_ratio=True),
        class_config(ResizeImages),
    ]

    test_preprocess_cfg = class_config(
        compose, transforms=preprocess_transforms
    )

    if normalize_image:
        batch_transforms = [
            class_config(NormalizeImages),
            class_config(PadImages),
        ]
    else:
        batch_transforms = [class_config(PadImages, value=114.0)]
    test_batchprocess_cfg = class_config(
        compose, transforms=batch_transforms + [class_config(ToTensor)]
    )

    test_dataset_cfg = class_config(
        DataPipe, datasets=test_dataset, preprocess_fn=test_preprocess_cfg
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
    image_size: tuple[int, int] = (800, 1440),
    normalize_image: bool = False,
    samples_per_gpu: int = 2,
    workers_per_gpu: int = 2,
) -> DataConfig:
    """Get the default config for BDD100K tracking."""
    data = DataConfig()

    data.train_dataloader = get_train_dataloader(
        data_backend=data_backend,
        image_size=image_size,
        normalize_image=normalize_image,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
    )

    data.test_dataloader = get_test_dataloader(
        data_backend=data_backend,
        image_size=image_size,
        normalize_image=normalize_image,
        samples_per_gpu=1,
        workers_per_gpu=1,
    )

    return data
