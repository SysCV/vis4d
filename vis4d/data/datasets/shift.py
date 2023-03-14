"""SHIFT dataset."""
from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor

from vis4d.common.imports import SCALABEL_AVAILABLE
from vis4d.data.const import CommonKeys as K
from vis4d.data.datasets.base import Dataset
from vis4d.data.datasets.util import filter_by_keys, im_decode, npy_decode
from vis4d.data.io import DataBackend, FileBackend, HDF5Backend, ZipBackend
from vis4d.data.typing import DictData

from .scalabel import ScalabelVideo

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
    "unlabeled": 0,
    "building": 1,
    "fence": 2,
    "other": 3,
    "pedestrian": 4,
    "pole": 5,
    "road line": 6,
    "road": 7,
    "sidewalk": 8,
    "vegetation": 9,
    "vehicle": 10,
    "wall": 11,
    "traffic sign": 12,
    "sky": 13,
    "ground": 14,
    "bridge": 15,
    "rail track": 16,
    "guard rail": 17,
    "traffic light": 18,
    "static": 19,
    "dynamic": 20,
    "water": 21,
    "terrain": 22,
}

if SCALABEL_AVAILABLE:
    from scalabel.label.io import load
    from scalabel.label.typing import Dataset as ScalabelData


def _get_extension(backend: DataBackend) -> str:
    """Get the appropriate file extension for the given backend."""
    if isinstance(backend, HDF5Backend):
        return ".hdf5"
    if isinstance(backend, ZipBackend):
        return ".zip"
    if isinstance(backend, FileBackend):  # pragma: no cover
        return ""
    raise ValueError(f"Unsupported backend {backend}.")  # pragma: no cover


class _SHIFTScalabelLabels(ScalabelVideo):
    """Helper class for labels in SHIFT that are stored in Scalabel format."""

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
        keys_to_load: Sequence[str],
        data_file: str = "",
        annotation_file: str = "",
        view: str = "front",
        backend: DataBackend = HDF5Backend(),
    ) -> None:
        """Initialize SHIFT dataset for one view.

        Args:
            data_root (str): Path to the root directory of the dataset.
            split (str): Which data split to load.
            data_file (str): Path to the data archive file. Default: "".
            keys_to_load (Sequence[str]): List of keys to load.
            annotation_file (str): Path to the annotation file. Default: "".
            view (str): Which view to load. Default: "front".
            backend (DataBackend): Backend to use for loading data. Default:
                HDF5Backend().
        """
        # Validate input
        assert split in set(
            ("train", "val", "test")
        ), f"Invalid split '{split}'"
        assert view in _SHIFTScalabelLabels.VIEWS, f"Invalid view '{view}'"

        # Set attributes
        annotation_path = os.path.join(
            data_root, "discrete", "images", split, view, annotation_file
        )
        ext = _get_extension(backend)
        data_path = os.path.join(
            data_root, "discrete", "images", split, view, f"{data_file}{ext}"
        )

        super().__init__(
            data_path,
            annotation_path,
            keys_to_load=keys_to_load,
            data_backend=backend,
        )

    def _generate_mapping(self) -> ScalabelData:
        """Generate data mapping."""
        # NOTE: Skipping validation for much faster loading
        return load(self.annotation_path, validate_frames=False)


class SHIFT(Dataset):
    """SHIFT dataset class, supporting multiple tasks and views."""

    DESCRIPTION = """SHIFT Dataset, a synthetic driving dataset for continuous
    multi-task domain adaptation"""
    HOMEPAGE = "https://www.vis.xyz/shift/"
    PAPER = "https://arxiv.org/abs/2206.08367"
    LICENSE = "CC BY-NC-SA 4.0"

    KEYS = [
        # Inputs
        K.images,
        K.original_hw,
        K.input_hw,
        K.points3d,
        # Scalabel formatted annotations
        K.intrinsics,
        K.extrinsics,
        K.timestamp,
        K.axis_mode,
        K.boxes2d,
        K.boxes2d_classes,
        K.boxes2d_track_ids,
        K.instance_masks,
        K.boxes3d,
        K.boxes3d_classes,
        K.boxes3d_track_ids,
        # Bit masks
        K.segmentation_masks,
        K.depth_maps,
        K.optical_flows,
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

    DATA_GROUPS = {
        "img": [
            K.images,
            K.original_hw,
            K.input_hw,
            K.intrinsics,
        ],
        "det_2d": [
            K.timestamp,
            K.axis_mode,
            K.extrinsics,
            K.boxes2d,
            K.boxes2d_classes,
            K.boxes2d_track_ids,
        ],
        "det_3d": [
            K.boxes3d,
            K.boxes3d_classes,
            K.boxes3d_track_ids,
        ],
        "det_insseg_2d": [
            K.instance_masks,
        ],
        "semseg": [
            K.segmentation_masks,
        ],
        "depth": [
            K.depth_maps,
        ],
        "flow": [
            K.optical_flows,
        ],
        "lidar": [
            K.points3d,
        ],
    }

    GROUPS_IN_SCALABEL = ["det_2d", "det_3d", "det_insseg_2d"]

    def __init__(
        self,
        data_root: str,
        split: str,
        keys_to_load: Sequence[str] = (K.images, K.boxes2d),
        views_to_load: Sequence[str] = ("front",),
        backend: DataBackend = HDF5Backend(),
    ) -> None:
        """Initialize SHIFT dataset."""
        # Validate input
        assert split in {"train", "val", "test"}, f"Invalid split '{split}'."
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
        if "det_2d" not in self._data_groups_to_load:
            raise ValueError(
                "In current implementation, the 'det_2d' data group must be"
                "loaded to load any other data group."
            )  # pragma: no cover

        self.scalabel_datasets = {}
        for view in self.views_to_load:
            if view == "center":
                # Load lidar data, only available for center view
                self.scalabel_datasets["center/lidar"] = _SHIFTScalabelLabels(
                    data_root=self.data_root,
                    split=self.split,
                    data_file="lidar",
                    annotation_file="det_3d.json",
                    view=view,
                    keys_to_load=(K.points3d, *self.DATA_GROUPS["det_3d"]),
                    backend=backend,
                )
            else:
                # Skip the lidar data group, which is loaded separately
                image_loaded = False
                for group in self._data_groups_to_load:
                    name = f"{view}/{group}"
                    keys_to_load = list(self.DATA_GROUPS[group])
                    # Load the image data group only once
                    if not image_loaded:
                        keys_to_load.extend(self.DATA_GROUPS["img"])
                        image_loaded = True

                    self.scalabel_datasets[name] = _SHIFTScalabelLabels(
                        data_root=self.data_root,
                        split=self.split,
                        data_file="img",
                        annotation_file=f"{group}.json",
                        view=view,
                        keys_to_load=keys_to_load,
                        backend=backend,
                    )

    def _get_data_groups(self, keys_to_load: Sequence[str]) -> list[str]:
        """Get the data groups that need to be loaded from Scalabel."""
        data_groups = []
        for data_group, group_keys in self.DATA_GROUPS.items():
            if data_group in self.GROUPS_IN_SCALABEL:
                # If the data group is loaded by Scalabel, add it to the list
                if any(key in group_keys for key in keys_to_load):
                    data_groups.append(data_group)
        return list(set(data_groups))

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
        raise ValueError(
            f"Invalid data group '{data_group}'"
        )  # pragma: no cover

    def _load_semseg(self, filepath: str) -> Tensor:
        """Load semantic segmentation data."""
        im_bytes = self.backend.get(filepath)
        image = im_decode(im_bytes)[..., 0]
        return torch.as_tensor(image, dtype=torch.int64).unsqueeze(0)

    def _load_depth(self, filepath: str, max_depth: float = 1000.0) -> Tensor:
        """Load depth data."""
        assert max_depth > 0, "Max depth value must be greater than 0."

        im_bytes = self.backend.get(filepath)
        image = im_decode(im_bytes)
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
        npy_bytes = self.backend.get(filepath)
        flow = npy_decode(npy_bytes, key="flow")
        return (
            torch.as_tensor(flow, dtype=torch.float32)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )

    def _get_frame_key(self, idx: int) -> tuple[str, str]:
        """Get the frame identifier (video name, frame name) by index."""
        if len(self.scalabel_datasets) > 0:
            frames = self.scalabel_datasets[
                list(self.scalabel_datasets.keys())[0]
            ].frames
            return frames[idx].videoName, frames[idx].name
        raise ValueError(
            "No Scalabel file has been loaded."
        )  # pragma: no cover

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        if len(self.scalabel_datasets) > 0:
            return len(
                self.scalabel_datasets[list(self.scalabel_datasets.keys())[0]]
            )
        raise ValueError(
            "No Scalabel file has been loaded."
        )  # pragma: no cover

    @property
    def video_to_indices(self) -> dict[str, list[int]]:
        """Group all dataset sample indices (int) by their video ID (str).

        Returns:
            dict[str, list[int]]: Mapping video to index.

        Raises:
            ValueError: If no Scalabel file has been loaded.
        """
        if len(self.scalabel_datasets) > 0:
            return self.scalabel_datasets[
                list(self.scalabel_datasets.keys())[0]
            ].video_to_indices
        raise ValueError(
            "No Scalabel file has been loaded."
        )  # pragma: no cover

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
                if K.points3d in self.keys_to_load:
                    data_dict_view.update(
                        self.scalabel_datasets["center/lidar"][idx]
                    )
            else:
                # Load data from Scalabel
                for group in self._data_groups_to_load:
                    data_dict_view.update(
                        self.scalabel_datasets[f"{view}/{group}"][idx]
                    )

                # Load data from bit masks
                if K.segmentation_masks in self.keys_to_load:
                    data_dict_view[K.segmentation_masks] = self._load(
                        view, "semseg", "png", video_name, frame_name
                    )
                if K.depth_maps in self.keys_to_load:
                    data_dict_view[K.depth_maps] = self._load(
                        view, "depth", "png", video_name, frame_name
                    )
                if K.optical_flows in self.keys_to_load:
                    data_dict_view[K.optical_flows] = self._load(
                        view, "flow", "npz", video_name, frame_name
                    )

            data_dict[view] = filter_by_keys(data_dict_view, self.keys_to_load)

        return data_dict
