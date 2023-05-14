"""SHIFT data loading config for data augmentation."""
from __future__ import annotations

from collections.abc import Sequence

from ml_collections.config_dict import ConfigDict

from vis4d.config.default.dataloader import get_dataloader_config
from vis4d.config.util import class_config
from vis4d.data.const import CommonKeys as K
from vis4d.data.datasets.shift import SHIFT
from vis4d.data.transforms.base import (
    RandomApply,
    Transform,
    compose,
    compose_batch,
)
from vis4d.data.transforms.crop import (
    CropBoxes2D,
    CropDepthMaps,
    CropImage,
    CropOpticalFlows,
    CropSegMasks,
    GenCropParameters,
)
from vis4d.data.transforms.flip import (
    FlipBoxes2D,
    FlipDepthMaps,
    FlipImage,
    FlipOpticalFlows,
    FlipSegMasks,
)
from vis4d.data.transforms.normalize import NormalizeImage
from vis4d.data.transforms.photometric import ColorJitter
from vis4d.data.transforms.resize import (
    GenerateResizeParameters,
    ResizeBoxes2D,
    ResizeDepthMaps,
    ResizeImage,
    ResizeOpticalFlows,
    ResizeSegMasks,
)
from vis4d.data.transforms.to_tensor import ToTensor


def get_train_preprocessing(
    image_size: tuple[int, int] = (800, 1280),
    crop_size: tuple[int, int] | None = None,
    horizontal_flip_prob: float = 0.5,
    color_jitter_prob: float = 0.0,
    keys_to_load: Sequence[str] = (K.images, K.seg_masks),
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

    Returns:
        The data preprocessing config.
    """
    for key_to_load in keys_to_load:
        assert key_to_load in SHIFT.KEYS, f"Invalid key: {key_to_load}"

    preprocess_transforms = []

    # Resize
    if image_size != (800, 1280):
        preprocess_transforms.append(
            class_config(
                GenerateResizeParameters,
                shape=image_size,
                keep_ratio=True,
            )
        )
        preprocess_transforms.append(class_config(ResizeImage))
        if K.seg_masks in keys_to_load:
            preprocess_transforms.append(class_config(ResizeSegMasks))
        if K.boxes2d in keys_to_load:
            preprocess_transforms.append(class_config(ResizeBoxes2D))
        if K.depth_maps in keys_to_load:
            preprocess_transforms.append(class_config(ResizeDepthMaps))
        if K.optical_flows in keys_to_load:
            preprocess_transforms.append(
                class_config(ResizeOpticalFlows, normalized_flow=False)
            )

    # Crop
    if crop_size is not None:
        preprocess_transforms.append(
            class_config(
                GenCropParameters, shape=crop_size, cat_max_ratio=0.75
            ),
        )
        preprocess_transforms.append(class_config(CropImage))
        if K.seg_masks in keys_to_load:
            preprocess_transforms.append(class_config(CropSegMasks))
        if K.boxes2d in keys_to_load:
            preprocess_transforms.append(class_config(CropBoxes2D))
        if K.depth_maps in keys_to_load:
            preprocess_transforms.append(class_config(CropDepthMaps))
        if K.optical_flows in keys_to_load:
            preprocess_transforms.append(class_config(CropOpticalFlows))

    # Random flip
    if horizontal_flip_prob > 0:
        flip_transforms = []
        flip_transforms.append(class_config(FlipImage))
        if K.seg_masks in keys_to_load:
            flip_transforms.append(class_config(FlipSegMasks))
        if K.boxes2d in keys_to_load:
            flip_transforms.append(class_config(FlipBoxes2D))
        if K.depth_maps in keys_to_load:
            flip_transforms.append(class_config(FlipDepthMaps))
        if K.optical_flows in keys_to_load:
            flip_transforms.append(class_config(FlipOpticalFlows))
        preprocess_transforms.append(
            class_config(
                RandomApply,
                transforms=flip_transforms,
                probability=horizontal_flip_prob,
            )
        )

    if color_jitter_prob > 0:
        preprocess_transforms.append(
            class_config(
                RandomApply,
                transforms=[class_config(ColorJitter)],
                probability=color_jitter_prob,
            )
        )

    preprocess_transforms.append(class_config(NormalizeImage))

    train_preprocess_cfg = class_config(
        compose, transforms=preprocess_transforms
    )
    train_batchprocess_cfg = class_config(
        compose_batch,
        transforms=[
            class_config(ToTensor),
        ],
    )
    return train_preprocess_cfg, train_batchprocess_cfg


def get_test_preprocessing(
    image_size: tuple[int, int] = (800, 1280),
    keys_to_load: Sequence[str] = (K.images, K.seg_masks),
) -> ConfigDict:
    """Get the default data preprocessing for SHIFT dataset.

    Args:
        image_size: The image size to resize to. Defaults to (800, 1280).
        keys_to_load: The keys to load from the dataset. Defaults to
            (K.images, K.seg_masks).

    Returns:
        The data preprocessing config.
    """
    for key_to_load in keys_to_load:
        assert key_to_load in SHIFT.KEYS, f"Invalid key: {key_to_load}"

    preprocess_transforms = []

    # Resize
    if image_size != (800, 1280):
        preprocess_transforms.append(
            class_config(
                GenerateResizeParameters,
                shape=image_size,
                keep_ratio=True,
            )
        )
        preprocess_transforms.append(class_config(ResizeImage))
        if K.seg_masks in keys_to_load:
            preprocess_transforms.append(class_config(ResizeSegMasks))
        if K.boxes2d in keys_to_load:
            preprocess_transforms.append(class_config(ResizeBoxes2D))
        if K.depth_maps in keys_to_load:
            preprocess_transforms.append(class_config(ResizeDepthMaps))
        if K.optical_flows in keys_to_load:
            preprocess_transforms.append(class_config(ResizeOpticalFlows))

    preprocess_transforms.append(class_config(NormalizeImage))

    test_preprocess_cfg = class_config(
        compose, transforms=preprocess_transforms
    )
    test_batchprocess_cfg = class_config(
        compose_batch,
        transforms=[
            class_config(ToTensor),
        ],
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
) -> ConfigDict:
    """Get the default config for BDD100K segmentation."""
    data = ConfigDict()

    train_preprocess_cfg, train_batchprocess_cfg = get_train_preprocessing(
        keys_to_load=keys_to_load,
        image_size=image_size,
        crop_size=crop_size,
        horizontal_flip_prob=horizontal_flip_prob,
        color_jitter_prob=color_jitter_prob,
    )

    test_preprocess_cfg, test_batchprocess_cfg = get_test_preprocessing(
        keys_to_load=keys_to_load,
        image_size=image_size,
    )

    data.train_dataloader = get_dataloader_config(
        preprocess_cfg=train_preprocess_cfg,
        dataset_cfg=train_dataset_cfg,
        batchprocess_cfg=train_batchprocess_cfg,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        shuffle=True,
    )
    data.test_dataloader = get_dataloader_config(
        preprocess_cfg=test_preprocess_cfg,
        dataset_cfg=test_dataset_cfg,
        batchprocess_cfg=test_batchprocess_cfg,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        shuffle=False,
    )

    return data


def get_shift_config(
    data_root: str = "data/shift/images",
    train_split: str = "train",
    train_framerate: str = "images",
    train_shift_type: str = "discrete",
    train_views_to_load: Sequence[str] = ("front",),
    train_keys_to_load: Sequence[str] = (K.images, K.seg_masks),
    test_split: str = "val",
    test_framerate: str = "images",
    test_shift_type: str = "discrete",
    test_views_to_load: Sequence[str] = ("front",),
    test_keys_to_load: Sequence[str] = (K.images, K.seg_masks),
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
    )
