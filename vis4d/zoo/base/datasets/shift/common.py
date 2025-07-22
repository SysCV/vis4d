"""SHIFT data loading config for data augmentation."""

from __future__ import annotations

from collections.abc import Sequence

from ml_collections.config_dict import ConfigDict

from vis4d.config import class_config
from vis4d.data.const import CommonKeys as K
from vis4d.data.data_pipe import DataPipe
from vis4d.data.datasets.shift import SHIFT
from vis4d.data.loader import default_collate, multi_sensor_collate
from vis4d.data.transforms.base import RandomApply, compose
from vis4d.data.transforms.crop import (
    CropBoxes2D,
    CropDepthMaps,
    CropImages,
    CropOpticalFlows,
    CropSegMasks,
    GenCropParameters,
)
from vis4d.data.transforms.flip import (
    FlipBoxes2D,
    FlipDepthMaps,
    FlipImages,
    FlipInstanceMasks,
    FlipOpticalFlows,
    FlipSegMasks,
)
from vis4d.data.transforms.normalize import NormalizeImages
from vis4d.data.transforms.photometric import ColorJitter
from vis4d.data.transforms.resize import (
    GenResizeParameters,
    ResizeBoxes2D,
    ResizeDepthMaps,
    ResizeImages,
    ResizeInstanceMasks,
    ResizeOpticalFlows,
    ResizeSegMasks,
)
from vis4d.data.transforms.select_sensor import SelectSensor
from vis4d.data.transforms.to_tensor import ToTensor
from vis4d.zoo.base import (
    get_inference_dataloaders_cfg,
    get_train_dataloader_cfg,
)

IMAGE_MEAN = [122.884, 117.266, 110.287]
IMAGE_STD = [59.925, 59.466, 60.69]


def get_train_preprocessing(
    image_size: tuple[int, int] = (800, 1280),
    crop_size: tuple[int, int] | None = None,
    horizontal_flip_prob: float = 0.5,
    color_jitter_prob: float = 0.0,
    keys_to_load: Sequence[str] = (K.images, K.seg_masks),
    views_to_load: Sequence[str] = ("front",),
) -> ConfigDict:
    """Get the default data preprocessing for SHIFT dataset.

    Args:
        image_size: The image size to resize to. Defaults to (800, 1280).
        crop_size: The crop size to crop to randomly, if not None. Defaults to
            None. This step is applied after the resize step.
        horizontal_flip_prob: The probability of horizontal flipping. Defaults
            to 0.5.
        color_jitter_prob: The probability of color jittering. Defaults to 0.5.
        keys_to_load: The keys to load from the dataset. Defaults to
            (K.images, K.seg_masks).
        views_to_load: The views to load from the dataset. Defaults to
            ("front",).

    Returns:
        The data preprocessing config.
    """
    preprocess_transforms = []

    for key_to_load in keys_to_load:
        assert key_to_load in SHIFT.KEYS, f"Invalid key: {key_to_load}"

    views_arg = {}
    if len(views_to_load) == 1:
        preprocess_transforms.append(
            class_config(
                SelectSensor,
                selected_sensor=views_to_load[0],
                sensors=views_to_load,
            )
        )
    elif len(views_to_load) > 1:
        views_arg["sensors"] = views_to_load

    # Resize
    if image_size != (800, 1280):
        preprocess_transforms.append(
            class_config(
                GenResizeParameters,
                shape=image_size,
                keep_ratio=True,
                **views_arg,
            )
        )
        preprocess_transforms.append(class_config(ResizeImages, **views_arg))
        if K.seg_masks in keys_to_load:
            preprocess_transforms.append(
                class_config(ResizeSegMasks, **views_arg)
            )
        if K.boxes2d in keys_to_load:
            preprocess_transforms.append(
                class_config(ResizeBoxes2D, **views_arg)
            )
        if K.instance_masks in keys_to_load:
            preprocess_transforms.append(
                class_config(ResizeInstanceMasks, **views_arg)
            )
        if K.depth_maps in keys_to_load:
            preprocess_transforms.append(
                class_config(ResizeDepthMaps, **views_arg)
            )
        if K.optical_flows in keys_to_load:
            preprocess_transforms.append(
                class_config(
                    ResizeOpticalFlows, normalized_flow=False, **views_arg
                )
            )

    # Crop
    if crop_size is not None:
        preprocess_transforms.append(
            class_config(
                GenCropParameters,
                shape=crop_size,
                cat_max_ratio=0.75,
                **views_arg,
            ),
        )
        preprocess_transforms.append(class_config(CropImages, **views_arg))
        if K.seg_masks in keys_to_load:
            preprocess_transforms.append(
                class_config(CropSegMasks, **views_arg)
            )
        if K.boxes2d in keys_to_load:
            preprocess_transforms.append(
                class_config(CropBoxes2D, **views_arg)
            )
        if K.depth_maps in keys_to_load:
            preprocess_transforms.append(
                class_config(CropDepthMaps, **views_arg)
            )
        if K.optical_flows in keys_to_load:
            preprocess_transforms.append(
                class_config(CropOpticalFlows, **views_arg)
            )

    # Random flip
    if horizontal_flip_prob > 0:
        flip_transforms = []
        flip_transforms.append(class_config(FlipImages))
        if K.seg_masks in keys_to_load:
            flip_transforms.append(class_config(FlipSegMasks))
        if K.boxes2d in keys_to_load:
            flip_transforms.append(class_config(FlipBoxes2D))
        if K.instance_masks in keys_to_load:
            flip_transforms.append(class_config(FlipInstanceMasks))
        if K.depth_maps in keys_to_load:
            flip_transforms.append(class_config(FlipDepthMaps))
        if K.optical_flows in keys_to_load:
            flip_transforms.append(class_config(FlipOpticalFlows))
        preprocess_transforms.append(
            class_config(
                RandomApply,
                transforms=flip_transforms,
                probability=horizontal_flip_prob,
                **views_arg,
            )
        )

    if color_jitter_prob > 0:
        preprocess_transforms.append(
            class_config(
                RandomApply,
                transforms=[class_config(ColorJitter, **views_arg)],
                probability=color_jitter_prob,
            )
        )

    preprocess_transforms.append(
        class_config(
            NormalizeImages, mean=IMAGE_MEAN, std=IMAGE_STD, **views_arg
        )
    )
    train_preprocess_cfg = class_config(
        compose, transforms=preprocess_transforms
    )

    batchprocess_transforms = [class_config(ToTensor, **views_arg)]
    train_batchprocess_cfg = class_config(
        compose, transforms=batchprocess_transforms
    )

    return train_preprocess_cfg, train_batchprocess_cfg


def get_test_preprocessing(
    image_size: tuple[int, int] = (800, 1280),
    keys_to_load: Sequence[str] = (K.images, K.seg_masks),
    views_to_load: Sequence[str] = ("front",),
) -> ConfigDict:
    """Get the default data preprocessing for SHIFT dataset.

    Args:
        image_size: The image size to resize to. Defaults to (800, 1280).
        keys_to_load: The keys to load from the dataset. Defaults to
            (K.images, K.seg_masks).
        views_to_load: The views to load from the dataset. Defaults to
            ("front",).

    Returns:
        The data preprocessing config.
    """
    preprocess_transforms = []

    for key_to_load in keys_to_load:
        assert key_to_load in SHIFT.KEYS, f"Invalid key: {key_to_load}"

    views_arg = {}
    if len(views_to_load) == 1:
        preprocess_transforms.append(
            class_config(
                SelectSensor,
                selected_sensor=views_to_load[0],
                sensors=views_to_load,
            )
        )
    elif len(views_to_load) > 1:
        views_arg["sensors"] = views_to_load

    # Resize
    if image_size != (800, 1280):
        preprocess_transforms.append(
            class_config(
                GenResizeParameters,
                shape=image_size,
                keep_ratio=True,
                **views_arg,
            )
        )
        preprocess_transforms.append(class_config(ResizeImages, **views_arg))
        if K.seg_masks in keys_to_load:
            preprocess_transforms.append(
                class_config(ResizeSegMasks, **views_arg)
            )
        if K.boxes2d in keys_to_load:
            preprocess_transforms.append(
                class_config(ResizeBoxes2D, **views_arg)
            )
        if K.depth_maps in keys_to_load:
            preprocess_transforms.append(
                class_config(ResizeDepthMaps, **views_arg)
            )
        if K.optical_flows in keys_to_load:
            preprocess_transforms.append(
                class_config(ResizeOpticalFlows, **views_arg)
            )

    preprocess_transforms.append(
        class_config(
            NormalizeImages, mean=IMAGE_MEAN, std=IMAGE_STD, **views_arg
        )
    )
    test_preprocess_cfg = class_config(
        compose, transforms=preprocess_transforms
    )

    batchprocess_transforms = [class_config(ToTensor, **views_arg)]

    test_batchprocess_cfg = class_config(
        compose, transforms=batchprocess_transforms
    )

    return test_preprocess_cfg, test_batchprocess_cfg


def get_shift_dataloader_config(
    train_dataset_cfg: ConfigDict,
    test_dataset_cfg: ConfigDict,
    keys_to_load: Sequence[str] = (K.images, K.seg_masks),
    image_size: tuple[int, int] = (800, 1280),
    crop_size: tuple[int, int] | None = None,
    horizontal_flip_prob: float = 0.5,
    color_jitter_prob: float = 0.5,
    samples_per_gpu: int = 2,
    workers_per_gpu: int = 2,
    train_views_to_load: Sequence[str] = ("front",),
    test_views_to_load: Sequence[str] = ("front",),
) -> ConfigDict:
    """Get the default config for BDD100K segmentation."""
    data = ConfigDict()

    train_preprocess_cfg, train_batchprocess_cfg = get_train_preprocessing(
        keys_to_load=keys_to_load,
        image_size=image_size,
        crop_size=crop_size,
        horizontal_flip_prob=horizontal_flip_prob,
        color_jitter_prob=color_jitter_prob,
        views_to_load=train_views_to_load,
    )

    test_preprocess_cfg, test_batchprocess_cfg = get_test_preprocessing(
        keys_to_load=keys_to_load,
        image_size=image_size,
        views_to_load=test_views_to_load,
    )

    data.train_dataloader = get_train_dataloader_cfg(
        preprocess_cfg=train_preprocess_cfg,
        dataset_cfg=train_dataset_cfg,
        batchprocess_cfg=train_batchprocess_cfg,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        shuffle=True,
        collate_fn=(
            multi_sensor_collate
            if len(train_views_to_load) > 1
            else default_collate
        ),
    )

    # Test Dataset Config
    test_dataset_cfg = class_config(
        DataPipe, datasets=test_dataset_cfg, preprocess_fn=test_preprocess_cfg
    )
    data.test_dataloader = get_inference_dataloaders_cfg(
        datasets_cfg=test_dataset_cfg,
        batchprocess_cfg=test_batchprocess_cfg,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        collate_fn=(
            multi_sensor_collate
            if len(test_views_to_load) > 1
            else default_collate
        ),
    )
    return data


def get_shift_config(  # pylint: disable=too-many-arguments, too-many-positional-arguments, line-too-long
    data_root: str = "data/shift/images",
    train_split: str = "train",
    train_framerate: str = "images",
    train_shift_type: str = "discrete",
    train_views_to_load: Sequence[str] = ("front",),
    train_keys_to_load: Sequence[str] = (K.images, K.seg_masks),
    train_attributes_to_load: Sequence[dict[str, str | float]] | None = None,
    train_skip_empty_frames: bool = False,
    test_split: str = "val",
    test_framerate: str = "images",
    test_shift_type: str = "discrete",
    test_views_to_load: Sequence[str] = ("front",),
    test_keys_to_load: Sequence[str] = (K.images, K.seg_masks),
    test_attributes_to_load: Sequence[dict[str, str | float]] | None = None,
    test_skip_empty_frames: bool = False,
    data_backend: None | ConfigDict = None,
    image_size: tuple[int, int] = (800, 1280),
    crop_size: tuple[int, int] | None = None,
    horizontal_flip_prob: float = 0.5,
    color_jitter_prob: float = 0.0,
    samples_per_gpu: int = 2,
    workers_per_gpu: int = 2,
) -> ConfigDict:
    """Get the default config for BDD100K segmentation."""
    train_dataset_cfg = class_config(
        SHIFT,
        data_root=data_root,
        split=train_split,
        framerate=train_framerate,
        shift_type=train_shift_type,
        views_to_load=train_views_to_load,
        keys_to_load=train_keys_to_load,
        attributes_to_load=train_attributes_to_load,
        skip_empty_frames=train_skip_empty_frames,
        backend=data_backend,
    )
    test_dataset_cfg = class_config(
        SHIFT,
        data_root=data_root,
        split=test_split,
        framerate=test_framerate,
        shift_type=test_shift_type,
        views_to_load=test_views_to_load,
        keys_to_load=test_keys_to_load,
        attributes_to_load=test_attributes_to_load,
        skip_empty_frames=test_skip_empty_frames,
        backend=data_backend,
    )

    return get_shift_dataloader_config(
        train_dataset_cfg=train_dataset_cfg,
        test_dataset_cfg=test_dataset_cfg,
        keys_to_load=train_keys_to_load,
        image_size=image_size,
        crop_size=crop_size,
        horizontal_flip_prob=horizontal_flip_prob,
        color_jitter_prob=color_jitter_prob,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        train_views_to_load=train_views_to_load,
        test_views_to_load=test_views_to_load,
    )
