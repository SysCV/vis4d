"""Default data loading config for object detection."""
from __future__ import annotations

from typing import Iterable

from ml_collections import FieldReference
from ml_collections.config_dict import ConfigDict

from vis4d.config.util import class_config


def det_augmentations() -> Iterable[ConfigDict]:
    """Returns the default image augmentations used for detection tasks.

    These augmentations consist of solely of left-right flipping the image and
    boxes.

    Returns:
        list[ConfigDict]: List with all transformations encoded as ConfigDict.
    """
    return (
        class_config("vis4d.data.transforms.flip.flip_image"),
        class_config("vis4d.data.transforms.flip.flip_boxes2d"),
    )


def det_preprocessing(
    target_img_height: int | FieldReference,
    target_img_width: int | FieldReference,
    augment_probability: float | FieldReference,
    augmentation_transforms: Iterable[ConfigDict] = det_augmentations(),
) -> ConfigDict:
    """Creates the default image preprocessing pipeling for a detection tasks.

    The pipeline consists of the following:
    1. Scale image and boxes to target size.
    2. Randomly apply given augmentations with the specified probability.
    3. Normalize the Image.

    Use this in combination with a dataset config to create a dataloader.

    Example:
    >>> preprocess_cfg = get_default_detection_preprocessing(480, 640, 0.5)
    >>> dataset_cfg = class_config("your.dataset.Dataset", root = "data/set/")
    >>> dataloader = get_dataloader_config(preprocess_cfg, dataset_cfg)

    Args:
        target_img_height (int | FieldReference): Target image height which
            should be fed to the network.
        target_img_width (int | FieldReference): Target image width which
            should be fed to the network.
        augment_probability (float | FieldReference): Probability to apply
            the augmentation operations.
        augmentation_transforms (list[ConfigDict], optional): List of
            transformation configurations that will be chained together and
            executed with propability `augmentation_probability`.

    Returns:
        ConfigDict: A ConfigDict that can be instantiated as
            vis4d.data.transforms.base.compose and passed to a dataloader.
    """
    transforms = [
        class_config(
            "vis4d.data.transforms.resize.resize_image",
            shape=(
                target_img_height,
                target_img_width,
            ),
            keep_ratio=True,
        ),
        class_config("vis4d.data.transforms.resize.resize_boxes2d"),
        class_config(
            "vis4d.data.transforms.base.random_apply",
            transforms=augmentation_transforms,
            probability=augment_probability,
        ),
        class_config("vis4d.data.transforms.normalize.normalize_image"),
    ]

    return class_config(
        "vis4d.data.transforms.base.compose",
        transforms=transforms,
    )
