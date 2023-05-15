"""SHIFT data loading config for segmentation."""
from __future__ import annotations

from ml_collections.config_dict import ConfigDict

from vis4d.common.typing import ArgsType
from vis4d.config.base.datasets.shift.common import get_shift_config
from vis4d.data.const import CommonKeys as K


def get_shift_segmentation_config(**kwargs: ArgsType) -> ConfigDict:
    """Get the config for the SHIFT segmentation task."""
    keys_to_load = (K.images, K.input_hw, K.original_hw, K.seg_masks)
    cfg = get_shift_config(
        train_keys_to_load=keys_to_load,
        test_keys_to_load=keys_to_load,
        horizontal_flip_prob=0.5,
        color_jitter_prob=0.5,
        crop_size=(512, 512),
        **kwargs,
    )
    return cfg


def get_shift_detection_config(**kwargs: ArgsType) -> ConfigDict:
    """Get the config for the SHIFT detection task."""
    keys_to_load = (
        K.images,
        K.input_hw,
        K.original_hw,
        K.boxes2d,
        K.boxes2d_classes,
    )
    cfg = get_shift_config(
        train_keys_to_load=keys_to_load,
        test_keys_to_load=keys_to_load,
        train_skip_empty_frames=True,
        test_skip_empty_frames=False,
        horizontal_flip_prob=0.5,
        color_jitter_prob=0.5,
        crop_size=None,
        **kwargs,
    )
    return cfg


def get_shift_instance_segmentation_config(**kwargs: ArgsType) -> ConfigDict:
    """Get the config for the SHIFT instance segmentation task."""
    keys_to_load = (
        K.images,
        K.input_hw,
        K.original_hw,
        K.boxes2d,
        K.boxes2d_classes,
        K.instance_masks,
    )
    cfg = get_shift_config(
        train_keys_to_load=keys_to_load,
        test_keys_to_load=keys_to_load,
        train_skip_empty_frames=True,
        test_skip_empty_frames=False,
        horizontal_flip_prob=0.5,
        color_jitter_prob=0.5,
        crop_size=None,
        **kwargs,
    )
    return cfg


def get_shift_depth_estimation_config(**kwargs: ArgsType) -> ConfigDict:
    """Get the config for the SHIFT depth estimation task."""
    keys_to_load = (K.images, K.input_hw, K.original_hw, K.depth_maps)
    cfg = get_shift_config(
        train_keys_to_load=keys_to_load,
        test_keys_to_load=keys_to_load,
        horizontal_flip_prob=0.5,
        color_jitter_prob=0.5,
        crop_size=None,
        **kwargs,
    )
    return cfg


def get_shift_optical_flow_config(**kwargs: ArgsType) -> ConfigDict:
    """Get the config for the SHIFT optical flow task."""
    keys_to_load = (K.images, K.input_hw, K.original_hw, K.optical_flows)
    cfg = get_shift_config(
        train_keys_to_load=keys_to_load,
        test_keys_to_load=keys_to_load,
        horizontal_flip_prob=0.5,
        color_jitter_prob=0.5,
        crop_size=None,
        **kwargs,
    )
    return cfg


def get_shift_tracking_config(**kwargs: ArgsType) -> ConfigDict:
    """Get the config for the SHIFT tracking task."""
    keys_to_load = (
        K.images,
        K.input_hw,
        K.original_hw,
        K.boxes2d,
        K.boxes2d_classes,
        K.boxes2d_track_ids,
    )
    cfg = get_shift_config(
        train_keys_to_load=keys_to_load,
        test_keys_to_load=keys_to_load,
        horizontal_flip_prob=0.5,
        color_jitter_prob=0.5,
        crop_size=None,
        **kwargs,
    )
    return cfg


def get_shift_multitask_2d_config(**kwargs: ArgsType) -> ConfigDict:
    """Get the config for the SHIFT multitask 2D task."""
    keys_to_load = (
        K.images,
        K.input_hw,
        K.original_hw,
        K.intrinsics,
        K.boxes2d,
        K.boxes2d_classes,
        K.boxes2d_track_ids,
        K.seg_masks,
        K.depth_maps,
    )
    cfg = get_shift_config(
        train_keys_to_load=keys_to_load,
        test_keys_to_load=keys_to_load,
        horizontal_flip_prob=0.5,
        color_jitter_prob=0.5,
        crop_size=None,
        **kwargs,
    )
    return cfg


def get_shift_multitask_3d_config(**kwargs: ArgsType) -> ConfigDict:
    """Get the config for the SHIFT multitask 3D task."""
    keys_to_load = (
        K.images,
        K.input_hw,
        K.original_hw,
        K.intrinsics,
        K.boxes2d,
        K.boxes2d_classes,
        K.boxes2d_track_ids,
        K.boxes3d,
        K.boxes3d_classes,
        K.boxes3d_track_ids,
        K.seg_masks,
        K.depth_maps,
    )
    cfg = get_shift_config(
        train_keys_to_load=keys_to_load,
        test_keys_to_load=keys_to_load,
        horizontal_flip_prob=0.5,
        color_jitter_prob=0.5,
        crop_size=None,
        **kwargs,
    )
    return cfg
