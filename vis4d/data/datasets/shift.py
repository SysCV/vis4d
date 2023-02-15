"""SHIFT dataset."""
from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor

from vis4d.common.imports import SCALABEL_AVAILABLE
from vis4d.data.const import CommonKeys
from vis4d.data.datasets.base import Dataset
from vis4d.data.datasets.util import filter_by_keys, im_decode, ply_decode
from vis4d.data.io import DataBackend, HDF5Backend
from vis4d.data.typing import DictData, DictStrAny

from .scalabel import Scalabel

shift_det_map = {
    "pedestrian": 0,
    "car": 1,
    "truck": 2,
    "bus": 3,
    "motorcycle": 4,
    "bicycle": 5,
}
shfit_track_map = {
    "pedestrian": 0,
    "car": 1,
    "truck": 2,
    "bus": 3,
    "motorcycle": 4,
    "bicycle": 5,
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


def _get_extension(backend: DataBackend):
    """Get the appropriate file extension for the given backend."""
    if isinstance(backend, HDF5Backend):
        return ".hdf5"
    return ""


def _get_data_archive(data_group: str) -> str:
    """Get the name of the data archive."""
    if data_group == "center":
        return "lidar"
    return "img"


class _SHIFTScalabelLabels(Scalabel):
    """Dataset class for the labels in SHIFT Dataset that are stored using
    Scalabel format."""

    DATA_GROUP = ["det_2d", "det_3d", "det_insseg_2d"]

    VIEWS = [
        "front",
        "center",
        "left_45",
        "left_90",
        "right_45",
        "right_90",
        "left_stereo",
    ]

    def __init__(
        self,
        data_root: str,
        split: str,
        data_group: str = "det_2d",
        view: str = "front",
        backend: DataBackend = HDF5Backend(),
        **kwargs: DictStrAny,
    ) -> None:
        """Initialize SHIFT dataset for one view.

        Args:
            data_root (str): Path to the root directory of the dataset.
            split (str): Which data split to load.
            data_group (str): Which data group to load. Default: "det_2d".
            view_to_load (str): Which view to load. Default: "front".
        """
        # Validate input
        assert split in set("train", "val", "test"), f"Invalid split '{split}'"
        assert view in _SHIFTScalabelLabels.VIEWS, f"Invalid view '{view}'"
        assert (
            data_group in _SHIFTScalabelLabels.DATA_GROUP
        ), f"Invalid data group of {data_group}."

        # Set attributes
        annotation_path = os.path.join(
            data_root, "discrete", "images", split, view, f"{data_group}.json"
        )
        data_arch = _get_data_archive(data_group)
        ext = _get_extension(backend)
        data_path = os.path.join(
            data_root, "discrete", "images", split, view, f"{data_arch}{ext}"
        )

        super().__init__(
            data_path, annotation_path, data_backend=backend, **kwargs
        )

    def _generate_mapping(self) -> ScalabelData:
        """Generate data mapping."""
        # NOTE: Skipping validation for much faster loading
        return load(self.annotation_path, validate_frames=False)


class SHIFT(Dataset):
    """SHIFT dataset."""

    DESCRIPTION = """SHIFT Dataset, a synthetic driving dataset for continuous
    multi-task domain adaptation"""
    PAPER = "https://arxiv.org/abs/2206.08367"
    HOMEPAGE = "https://www.vis.xyz/shift/"
    LICENSE = "CC BY-NC-SA 4.0"

    KEYS = [
        # Scalabel formatted annotations
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
        CommonKeys.masks,
        CommonKeys.boxes3d,
        CommonKeys.boxes3d_classes,
        CommonKeys.boxes3d_track_ids,
        # Bit masks
        CommonKeys.segmentation_masks,
        CommonKeys.depth_maps,
        CommonKeys.optical_flows,
        # Point clouds
        CommonKeys.points3d,
    ]

    VIEWS = [
        "front",
        "center",
        "left_45",
        "left_90",
        "right_45",
        "right_90",
        "left_stereo",
    ]

    def __init__(
        self,
        data_root: str,
        split: str,
        keys_to_load: Sequence[str] = (CommonKeys.images, CommonKeys.boxes2d),
        views_to_load: Sequence[str] = ("front",),
        backend: DataBackend = HDF5Backend(),
    ) -> None:
        """Initialize SHIFT dataset."""
        # Validate input
        assert split in set("train", "val", "test"), f"Invalid split '{split}'"
        self.validate_keys(keys_to_load)

        # Set attributes
        self.data_root = data_root
        self.split = split
        self.keys_to_load = keys_to_load
        self.views_to_load = views_to_load
        self.backend = backend
        self.ext = _get_extension(backend)
        self.annotation_base = os.path.join(
            self.data_root, "discrete", "images", self.split
        )

        # Get the data groups' classes that need to be loaded
        self._data_groups_to_load = self._get_data_groups(keys_to_load)
        self.scalabel_datasets = {}
        for view in self.views_to_load:
            if view == "center":
                continue
            for group in self._data_groups_to_load:
                name = f"{view}/{group}"
                self.scalabel_datasets[name] = _SHIFTScalabelLabels(
                    data_root=self.data_root,
                    split=self.split,
                    data_group=group,
                    view=view,
                    keys_to_load=keys_to_load,
                    backend=backend,
                )

    def _get_data_groups(self, keys_to_load: Sequence[str]) -> list[str]:
        """Get the data groups that need to be loaded."""
        data_groups = []
        if any(
            key in keys_to_load
            for key in (
                CommonKeys.intrinsics,
                CommonKeys.extrinsics,
                CommonKeys.timestamp,
                CommonKeys.axis_mode,
                CommonKeys.boxes2d,
                CommonKeys.boxes2d_classes,
                CommonKeys.boxes2d_track_ids,
            )
        ):
            data_groups.append("det_2d")
        if any(
            key in keys_to_load
            for key in (
                CommonKeys.boxes3d,
                CommonKeys.boxes3d_classes,
                CommonKeys.boxes3d_track_ids,
            )
        ):
            data_groups.append("det_3d")
        if any(key in keys_to_load for key in (CommonKeys.masks,)):
            data_groups.append("det_insseg_2d")
        return data_groups

    def _load(
        self, view: str, data_group: str, file_ext: str, video: str, frame: str
    ) -> Tensor:
        """Load data from the given data group."""
        frame_number = frame.split("_")[0]
        filepath = os.path.join(
            self.annotation_base,
            view,
            f"{data_group}{self.ext}",
            video,
            f"{frame_number}_{data_group}_{view}.{file_ext}",
        )
        if data_group == "semseg":
            return self._load_semseg(filepath)
        if data_group == "depth":
            return self._load_depth(filepath)
        if data_group == "flow":
            return self._load_flow(filepath)
        if data_group == "lidar":
            return self._load_lidar(filepath)
        raise ValueError(f"Invalid data group '{data_group}'")

    def _load_semseg(self, filepath: str) -> Tensor:
        """Load semantic segmentation data."""
        im_bytes = self.backend.get(filepath)
        image = im_decode(im_bytes, mode="L")
        return torch.as_tensor(image, dtype=torch.int64)

    def _load_depth(self, filepath: str, max_depth: float = 1000.0) -> Tensor:
        """Load depth data."""
        assert max_depth > 0, "Max depth value must be greater than 0."

        im_bytes = self.backend.get(filepath)
        image = im_decode(im_bytes, mode="RGB")
        if image.shape[2] > 3:  # pragma: no cover
            image = image[:, :, :3]
        image = image.astype(np.float32)

        # Convert to depth
        depth = (
            image[:, :, 2] * 256 * 256 + image[:, :, 1] * 256 + image[:, :, 0]
        )
        return torch.as_tensor(
            np.ascontiguousarray(depth / max_depth),
            dtype=torch.float32,
        ).unsqueeze(0)

    def _load_flow(self, filepath: str) -> Tensor:
        """Load optical flow data."""
        raise NotImplementedError

    def _load_lidar(self, filepath: str) -> Tensor:
        """Load lidar data."""
        ply_bytes = self.backend.get(filepath)
        points = ply_decode(ply_bytes)
        return torch.as_tensor(points, dtype=torch.float32)

    def _get_frame_key(self, idx: int) -> tuple[str, str]:
        """Get the frame identifier (video name, frame name) by index."""
        if len(self.scalabel_datasets) > 0:
            frames = self.scalabel_datasets[
                list(self.scalabel_datasets.keys())[0]
            ].frames
            return frames[idx].videoName, frames[idx].name

        raise NotImplementedError

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        if len(self.scalabel_datasets) > 0:
            return len(
                self.scalabel_datasets[list(self.scalabel_datasets.keys())[0]]
            )

        raise NotImplementedError

    def __getitem__(self, idx: int) -> DictData:
        """Get single sample.

        Args:
            idx (int): Index of sample.

        Returns:
            DictData: sample at index in Vis4D input format.
        """

        # load camera frames
        data_dict = {}
        for view in self.views_to_load:
            data_dict_view = {}
            video_name, frame_name = self._get_frame_key(idx)

            if view == "center":
                # Lidar is only available in the center view
                if CommonKeys.points3d in self.keys_to_load:
                    data_dict_view[CommonKeys.points3d] = self._load(
                        view, "lidar", "ply", video_name, frame_name
                    )
            else:
                # Load data from Scalabel
                for group in self._data_groups_to_load:
                    data_dict_view.update(
                        self.scalabel_datasets[f"{view}/{group}"][idx]
                    )

                # Load data from bit masks
                if CommonKeys.segmentation_masks in self.keys_to_load:
                    data_dict_view[CommonKeys.segmentation_masks] = self._load(
                        view, "semseg", "png", video_name, frame_name
                    )
                if CommonKeys.depth_maps in self.keys_to_load:
                    data_dict_view[CommonKeys.depth_maps] = self._load(
                        view, "depth", "png", video_name, frame_name
                    )
                if CommonKeys.optical_flows in self.keys_to_load:
                    data_dict_view[CommonKeys.optical_flows] = self._load(
                        view, "flow", "npz", video_name, frame_name
                    )

            data_dict[view] = filter_by_keys(data_dict_view, self.keys_to_load)

        return data_dict
