"""Default data loading config for 3D segmentation."""
from __future__ import annotations

from typing import Sequence

from ml_collections import FieldReference
from ml_collections.config_dict import ConfigDict

from vis4d.config.util import class_config


def default_3d_augmentation() -> Sequence[ConfigDict]:
    """Configuration for default augmentation operations.

    This functions returns the default 3D augmentation which are used
    for pointclouds.

    It consists of:W
    - 'add_norm_noise': Adding normal distributed noise.

    Returns:
        Sequence[ConfigDict]: A list of configuration objects for different
        transform operators.
    """
    transforms = (
        class_config(
            "vis4d.data.transforms.points.add_norm_noise",
            std=0.01,
        ),
    )
    return transforms


def default_3d_preprocessing(
    keys_to_load: list[str] | FieldReference,
    augment_probability: float | FieldReference,
    augment_transforms: Sequence[ConfigDict] = default_3d_augmentation(),
) -> ConfigDict:
    """Default Pipeline to preprocess 3D data.

    This pipeline consists of the following operations:
    1. Extract the boundaries of the pointcloud.
    2. Sample points from a random block.
    3. Randomly apply the provided augmentations.
    4. Center and normalize the pointcloud.
    5. Flip last two channels from [n_pts, ?] to [?, n_pts]
    6. Concatenate all features into one feature tensor by stacking them
        along the first axis.

    Args:
        keys_to_load (list[str] | FieldReference): Which keys should be loaded
            from the dataset. The first key must match the xyz coordiantes of
            the points.
        augment_probability (float | FieldReference): Changes to apply the
            augmentations.
        augment_transforms (Sequence[ConfigDict], optional): List of
            augmentation transforms that should be applied.
            Defaults to default_3d_augmentation().

    Returns:
        ConfigDict: Configuration dict that can be passed to a dataloader.
    """
    transforms = [
        class_config("vis4d.data.transforms.points.extract_pc_bounds"),
        class_config(
            "vis4d.data.transforms.point_sampling.sample_points_block_random",
            in_keys=keys_to_load,
            out_keys=keys_to_load,
        ),
        class_config(
            "vis4d.data.transforms.base.random_apply",
            transforms=augment_transforms,
            probability=augment_probability,
        ),
        class_config(
            "vis4d.data.transforms.points.center_and_normalize",
            out_keys=("points3d_normalized",),
            normalize=False,
        ),
        class_config(
            "vis4d.data.transforms.points.move_pts_to_last_channel",
            in_keys=keys_to_load + ["points3d_normalized"],
            out_keys=keys_to_load + ["points3d_normalized"],
        ),
        class_config(
            "vis4d.data.transforms.points.concatenate_point_features",
            in_keys=keys_to_load + ["points3d_normalized"],
            out_keys="points3d",
        ),
    ]

    return class_config(
        "vis4d.data.transforms.base.compose",
        transforms=transforms,
    )
