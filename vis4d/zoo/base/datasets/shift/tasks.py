"""SHIFT data loading config for segmentation."""

from __future__ import annotations

from ml_collections.config_dict import ConfigDict

from vis4d.common.typing import ArgsType
from vis4d.data.const import CommonKeys as K
from vis4d.engine.connectors import data_key, pred_key

from .common import get_shift_config

CONN_SHIFT_DET_EVAL = {
    "frame_ids": data_key("frame_ids"),
    "sample_names": data_key("sample_names"),
    "sequence_names": data_key("sequence_names"),
    "pred_boxes": pred_key("boxes"),
    "pred_scores": pred_key("scores"),
    "pred_classes": pred_key("class_ids"),
}
CONN_SHIFT_INS_EVAL = {
    "frame_ids": data_key("frame_ids"),
    "sample_names": data_key("sample_names"),
    "sequence_names": data_key("sequence_names"),
    "pred_boxes": pred_key("boxes.boxes"),
    "pred_scores": pred_key("boxes.scores"),
    "pred_classes": pred_key("boxes.class_ids"),
    "pred_masks": pred_key("masks.masks"),
}


def get_shift_sem_seg_config(**kwargs: ArgsType) -> ConfigDict:
    """Get the config for the SHIFT segmentation task."""
    keys_to_load = (K.images, K.input_hw, K.original_hw, K.seg_masks)
    cfg = get_shift_config(
        train_keys_to_load=keys_to_load,
        test_keys_to_load=keys_to_load,
        horizontal_flip_prob=0.5,
        color_jitter_prob=0.5,
        crop_size=kwargs.get("crop_size", (512, 1024)),
        **kwargs,
    )
    return cfg


def get_shift_det_config(**kwargs: ArgsType) -> ConfigDict:
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
        color_jitter_prob=0.0,
        crop_size=kwargs.get("crop_size", None),
        **kwargs,
    )
    return cfg


def get_shift_instance_seg_config(**kwargs: ArgsType) -> ConfigDict:
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
        crop_size=kwargs.get("crop_size", None),
        **kwargs,
    )
    return cfg


def get_shift_depth_est_config(**kwargs: ArgsType) -> ConfigDict:
    """Get the config for the SHIFT depth estimation task."""
    keys_to_load = (K.images, K.input_hw, K.original_hw, K.depth_maps)
    cfg = get_shift_config(
        train_keys_to_load=keys_to_load,
        test_keys_to_load=keys_to_load,
        horizontal_flip_prob=0.5,
        crop_size=kwargs.get("crop_size", None),
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
        crop_size=kwargs.get("crop_size", None),
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
        crop_size=kwargs.get("crop_size", None),
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
        crop_size=kwargs.get("crop_size", None),
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
        crop_size=kwargs.get("crop_size", None),
        **kwargs,
    )
    return cfg
