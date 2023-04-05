"""Default data loading config for object detection."""
from __future__ import annotations

from collections.abc import Iterable

from ml_collections import FieldReference
from ml_collections.config_dict import ConfigDict

from vis4d.config.util import class_config
from vis4d.data.const import CommonKeys as K
from vis4d.data.transforms import RandomApply, compose, flip, normalize, resize


def segment_augmentations() -> Iterable[ConfigDict]:
    """Returns the default image augmentations used for detection tasks.

    These augmentations consist of solely of left-right flipping the image and
    boxes.

    Returns:
        list[ConfigDict]: List with all transformations encoded as ConfigDict.
    """
    return (
        class_config(
            flip.FlipImage, in_keys=(K.images,), out_keys=(K.images,)
        ),
        class_config(
            flip.FlipSemanticMasks,
            in_keys=(K.segmentation_masks,),
            out_keys=(K.segmentation_masks,),
        ),
    )


def segment_preprocessing(
    target_img_height: int | FieldReference,
    target_img_width: int | FieldReference,
    keep_ratio: bool | FieldReference,
    augment_probability: float | FieldReference,
    augmentation_transforms: Iterable[ConfigDict] = segment_augmentations(),
) -> ConfigDict:
    """Creates the default image preprocessing pipeling for a segment. tasks.

    The pipeline consists of the following:
    1. Scale image and masks to target size.
    2. Randomly apply given augmentations with the specified probability.
    3. Normalize the Image.

    Use this in combination with a dataset config to create a dataloader.

    Example:
    >>> preprocess_cfg = segment_preprocessing(480, 640, 0.5)
    >>> dataset_cfg = class_config("your.dataset.Dataset", root = "data/set/")
    >>> dataloader = get_dataloader_config(preprocess_cfg, dataset_cfg)

    Args:
        target_img_height (int | FieldReference): Target image height which
            should be fed to the network.
        target_img_width (int | FieldReference): Target image width which
            should be fed to the network.
        keep_ratio (bool | FieldReference): Whether to keep the aspect ratio.
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
            resize.GenerateResizeParameters,
            shape=(target_img_height, target_img_width),
            keep_ratio=keep_ratio,
        ),
        class_config(resize.ResizeImage),
        class_config(resize.ResizeSemanticMasks),
        class_config(
            RandomApply,
            transforms=augmentation_transforms,
            probability=augment_probability,
        ),
        class_config(normalize.NormalizeImage),
    ]

    return class_config(compose, transforms=transforms)
