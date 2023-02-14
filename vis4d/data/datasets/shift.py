"""SHIFT dataset."""
from __future__ import annotations

import os

from collections.abc import Sequence
from vis4d.common.imports import SCALABEL_AVAILABLE
from vis4d.data.const import CommonKeys

from .scalabel import Scalabel

shift_det_map = {
    "bicycle": 0,
    "car": 1,
    "motor": 2,
    "truck": 3,
}
shfit_track_map = {
    "bicycle": 0,
    "car": 1,
    "motor": 2,
    "truck": 3,
}
shift_seg_map = {
    "road": 0,
    "sidewalk": 1,
    "building": 2,
    "wall": 3,
    "fence": 4,
    "pole": 5,
    "traffic light": 6,
    "traffic sign": 7,
    "vegetation": 8,
    "terrain": 9,
    "sky": 10,
    "pedestrian": 11,
    "rider": 12,
    "car": 13,
    "truck": 14,
    "bus": 15,
    "train": 16,
    "motorcycle": 17,
    "bicycle": 18,
}

if SCALABEL_AVAILABLE:
    from scalabel.label.io import load
    from scalabel.label.typing import Dataset as ScalabelData


class SHIFT_ONE_VIEWS(Scalabel):
    """SHIFT dataset, based on Scalabel."""

    def __init__(
        self,
        data_root: str,
        split: str,
        keys_to_load: Sequence[str] = (CommonKeys.images, CommonKeys.boxes2d),
        view_to_load: str = "front",
    ) -> None:
        """Initialize SHIFT ONE VIEWS dataset."""
        self.annotation_path = os.path.join(
            self.data_root, "discrete", "", f"{self.split}.json"
        )

    def _generate_mapping(self) -> ScalabelData:
        """Generate data mapping."""
        shift_anns = load(self.annotation_path, validate_frames=False)
        return ScalabelData(frames=shift_anns.frames, config=shift_anns.config)


class SHIFT(Dataset):
    """SHIFT dataset, based on Scalabel."""

    KEYS = [
        CommonKeys.images,
        CommonKeys.original_hw,
        CommonKeys.input_hw,
        CommonKeys.intrinsics,
        CommonKeys.extrinsics,
        CommonKeys.timestamp,
        CommonKeys.axis_mode,
        CommonKeys.boxes2d,
        CommonKeys.boxes2d_classes,
        CommonKeys.boxes2d_track_ids,
        CommonKeys.boxes3d,
        CommonKeys.boxes3d_classes,
        CommonKeys.boxes3d_track_ids,
    ]

    def __init__(
        self,
        data_root: str,
        split: str,
        keys_to_load: Sequence[str] = (CommonKeys.images, CommonKeys.boxes2d),
        views_to_load: Sequence[str] = ("front",),
    ) -> None:
        """Initialize SHIFT dataset."""
        self.annotation_path = os.path.join(
            self.data_root, "annotations", f"{self.split}.json"
        )

    def _generate_mapping(self) -> ScalabelData:
        """Generate data mapping."""
        shift_anns = load(self.annotation_path)
        frames = bdd100k_anns.frames
        bdd100k_cfg = load_bdd100k_config(self.config_path)
        scalabel_frames = bdd100k_to_scalabel(frames, bdd100k_cfg)
        return ScalabelData(
            frames=scalabel_frames, config=bdd100k_cfg.scalabel
        )
