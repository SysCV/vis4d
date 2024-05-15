"""SHIFT dataset."""

from __future__ import annotations

import json
import multiprocessing
import os
from collections.abc import Sequence
from functools import partial

import numpy as np
from tqdm import tqdm

from vis4d.common.imports import SCALABEL_AVAILABLE
from vis4d.common.logging import rank_zero_info
from vis4d.common.typing import NDArrayF32, NDArrayI64, NDArrayNumber
from vis4d.data.const import CommonKeys as K
from vis4d.data.datasets.base import VideoDataset
from vis4d.data.datasets.util import im_decode, npy_decode
from vis4d.data.io import DataBackend, FileBackend, HDF5Backend, ZipBackend
from vis4d.data.typing import DictData

from .base import VideoDataset, VideoMapping
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
shift_seg_ignore = [
    "unlabeled",
    "other",
    "ground",
    "bridge",
    "rail track",
    "guard rail",
    "static",
    "dynamic",
    "water",
]

if SCALABEL_AVAILABLE:
    from scalabel.label.io import parse
    from scalabel.label.typing import Config
    from scalabel.label.typing import Dataset as ScalabelData
else:
    raise ImportError("scalabel is not installed.")


def _get_extension(backend: DataBackend) -> str:
    """Get the appropriate file extension for the given backend."""
    if isinstance(backend, HDF5Backend):
        return ".hdf5"
    if isinstance(backend, ZipBackend):
        return ".zip"
    if isinstance(backend, FileBackend):  # pragma: no cover
        return ""
    raise ValueError(f"Unsupported backend {backend}.")  # pragma: no cover


class _SHIFTScalabelLabels(Scalabel):
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
        data_file: str = "",
        keys_to_load: Sequence[str] = (K.images, K.boxes2d),
        attributes_to_load: Sequence[dict[str, str | float]] | None = None,
        annotation_file: str = "",
        view: str = "front",
        framerate: str = "images",
        shift_type: str = "discrete",
        skip_empty_frames: bool = False,
        backend: DataBackend = HDF5Backend(),
        verbose: bool = False,
        num_workers: int = 1,
    ) -> None:
        """Initialize SHIFT dataset for one view.

        Args:
            data_root (str): Path to the root directory of the dataset.
            split (str): Which data split to load.
            data_file (str): Path to the data archive file. Default: "".
            keys_to_load (Sequence[str]): List of keys to load.
                Default: (K.images, K.boxes2d).
            attributes_to_load (Sequence[dict[str, str | float]] | None):
                List of attributes to load. Default: None.
            annotation_file (str): Path to the annotation file. Default: "".
            view (str): Which view to load. Default: "front". Options: "front",
                "center", "left_45", "left_90", "right_45", "right_90", and
                "left_stereo".
            framerate (str): Which framerate to load. Default: "images".
            shift_type (str): Which shift type to load. Default: "discrete".
                Options: "discrete", "continuous/1x", "continuous/10x", and
                "continuous/100x".
            skip_empty_frames (bool): Whether to skip frames with no
                instance annotations. Default: False.
            backend (DataBackend): Backend to use for loading data. Default:
                HDF5Backend().
            verbose (bool): Whether to print verbose logs. Default: False.
            num_workers (int): Number of workers to use for loading data.
                Default: 1.
        """
        self.verbose = verbose
        self.num_workers = num_workers

        # Validate input
        assert split in {"train", "val", "test"}, f"Invalid split '{split}'"
        assert view in _SHIFTScalabelLabels.VIEWS, f"Invalid view '{view}'"

        # Set attributes
        ext = _get_extension(backend)
        if shift_type.startswith("continuous"):
            shift_speed = shift_type.split("/")[-1]
            annotation_path = os.path.join(
                data_root,
                "continuous",
                framerate,
                shift_speed,
                split,
                view,
                annotation_file,
            )
            data_path = os.path.join(
                data_root,
                "continuous",
                framerate,
                shift_speed,
                split,
                view,
                f"{data_file}{ext}",
            )
        else:
            annotation_path = os.path.join(
                data_root, "discrete", framerate, split, view, annotation_file
            )
            data_path = os.path.join(
                data_root,
                "discrete",
                framerate,
                split,
                view,
                f"{data_file}{ext}",
            )
        super().__init__(
            data_path,
            annotation_path,
            data_backend=backend,
            keys_to_load=keys_to_load,
            attributes_to_load=attributes_to_load,
            skip_empty_samples=skip_empty_frames,
        )

    def _generate_mapping(self) -> ScalabelData:
        """Generate data mapping."""
        # Skipping validation for much faster loading
        if self.verbose:
            rank_zero_info(
                "Loading annotation from '%s' ...", self.annotation_path
            )
        return self._load(self.annotation_path)

    def _load(self, filepath: str) -> ScalabelData:
        """Load labels from a json file or a folder of json files."""
        raw_frames: list[DictData] = []
        raw_groups: list[DictData] = []
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{filepath} does not exist.")

        def process_file(filepath: str) -> DictData | None:
            raw_cfg = None
            with open(filepath, mode="r", encoding="utf-8") as fp:
                content = json.load(fp)
            if isinstance(content, dict):
                raw_frames.extend(content["frames"])
                if "groups" in content and content["groups"] is not None:
                    raw_groups.extend(content["groups"])
                if "config" in content and content["config"] is not None:
                    raw_cfg = content["config"]
            elif isinstance(content, list):
                raw_frames.extend(content)
            else:
                raise TypeError(
                    "The input file contains neither dict nor list."
                )

            rank_zero_info(
                "Loading SHIFT annotation from '%s' Done.", filepath
            )
            return raw_cfg

        cfg = None
        if os.path.isfile(filepath) and filepath.endswith("json"):
            ret_cfg = process_file(filepath)
            if ret_cfg is not None:
                cfg = ret_cfg
        else:
            raise TypeError("Inputs must be a folder or a JSON file.")

        config = None
        if cfg is not None:
            config = Config(**cfg)

        parse_func = partial(parse, validate_frames=False)
        if self.num_workers > 1:
            with multiprocessing.Pool(self.num_workers) as pool:
                frames = []
                with tqdm(total=len(raw_frames)) as pbar:
                    for result in pool.imap_unordered(
                        parse_func, raw_frames, chunksize=1000
                    ):
                        frames.append(result)
                        pbar.update()
        else:
            frames = [parse_func(frame) for frame in raw_frames]
        return ScalabelData(frames=frames, config=config, groups=None)


class SHIFT(VideoDataset):
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
        K.seg_masks,
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
            K.seg_masks,
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
        attributes_to_load: Sequence[dict[str, str | float]] | None = None,
        framerate: str = "images",
        shift_type: str = "discrete",
        skip_empty_frames: bool = False,
        backend: DataBackend = HDF5Backend(),
        num_workers: int = 1,
        verbose: bool = False,
    ) -> None:
        """Initialize SHIFT dataset."""
        super().__init__(data_backend=backend)
        # Validate input
        assert split in {"train", "val", "test"}, f"Invalid split '{split}'."
        assert framerate in {
            "images",
            "videos",
        }, f"Invalid framerate '{framerate}'. Must be 'images' or 'videos'."
        assert shift_type in {
            "discrete",
            "continuous/1x",
            "continuous/10x",
            "continuous/100x",
        }, (
            f"Invalid shift_type '{shift_type}'. "
            "Must be one of 'discrete', 'continuous/1x', 'continuous/10x', "
            "or 'continuous/100x'."
        )
        self.validate_keys(keys_to_load)

        # Set attributes
        self.data_root = data_root
        self.split = split
        self.keys_to_load = keys_to_load
        self.views_to_load = views_to_load
        self.attributes_to_load = attributes_to_load
        self.framerate = framerate
        self.shift_type = shift_type
        self.backend = backend
        self.verbose = verbose
        self.ext = _get_extension(backend)
        if self.shift_type.startswith("continuous"):
            shift_speed = self.shift_type.split("/")[-1]
            self.annotation_base = os.path.join(
                self.data_root,
                "continuous",
                self.framerate,
                shift_speed,
                self.split,
            )
        else:
            self.annotation_base = os.path.join(
                self.data_root, self.shift_type, self.framerate, self.split
            )
        if self.verbose:
            print(f"Base: {self.annotation_base}. Backend: {self.backend}")

        # Get the data groups' classes that need to be loaded
        self._data_groups_to_load = self._get_data_groups(keys_to_load)
        if "det_2d" not in self._data_groups_to_load:
            raise ValueError(
                "In current implementation, the 'det_2d' data group must be "
                "loaded to load any other data group."
            )

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
                    framerate=self.framerate,
                    shift_type=self.shift_type,
                    keys_to_load=(K.points3d, *self.DATA_GROUPS["det_3d"]),
                    attributes_to_load=self.attributes_to_load,
                    skip_empty_frames=skip_empty_frames,
                    backend=backend,
                    num_workers=num_workers,
                    verbose=verbose,
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
                        framerate=self.framerate,
                        shift_type=self.shift_type,
                        keys_to_load=keys_to_load,
                        attributes_to_load=self.attributes_to_load,
                        skip_empty_frames=skip_empty_frames,
                        backend=backend,
                        num_workers=num_workers,
                        verbose=verbose,
                    )

        self.video_mapping = self._generate_video_mapping()

    def validate_keys(self, keys_to_load: Sequence[str]) -> None:
        """Validate that all keys to load are supported."""
        for k in keys_to_load:
            if k not in self.KEYS:
                raise ValueError(f"Key '{k}' is not supported!")

    def _get_data_groups(self, keys_to_load: Sequence[str]) -> list[str]:
        """Get the data groups that need to be loaded from Scalabel."""
        data_groups = ["det_2d"]
        for data_group, group_keys in self.DATA_GROUPS.items():
            if data_group in self.GROUPS_IN_SCALABEL:
                # If the data group is loaded by Scalabel, add it to the list
                if any(key in group_keys for key in keys_to_load):
                    data_groups.append(data_group)
        return list(set(data_groups))

    def _load(
        self, view: str, data_group: str, file_ext: str, video: str, frame: str
    ) -> NDArrayNumber:
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

    def _load_semseg(self, filepath: str) -> NDArrayI64:
        """Load semantic segmentation data."""
        im_bytes = self.backend.get(filepath)
        image = im_decode(im_bytes)[..., 0]
        return image.astype(np.int64)

    def _load_depth(
        self, filepath: str, depth_factor: float = 16777.216  # 256 ^ 3 / 1000
    ) -> NDArrayF32:
        """Load depth data."""
        assert depth_factor > 0, "Max depth value must be greater than 0."

        im_bytes = self.backend.get(filepath)
        image = im_decode(im_bytes)
        if image.shape[2] > 3:  # pragma: no cover
            image = image[:, :, :3]
        image = image.astype(np.float32)

        # Convert to depth
        depth = (
            image[:, :, 2] * 256 * 256 + image[:, :, 1] * 256 + image[:, :, 0]
        )
        return np.ascontiguousarray(depth / depth_factor, dtype=np.float32)

    def _load_flow(self, filepath: str) -> NDArrayF32:
        """Load optical flow data."""
        npy_bytes = self.backend.get(filepath)
        flow = npy_decode(npy_bytes, key="flow")
        flow = flow[:, :, [1, 0]]  # Convert to (u, v) format
        flow *= flow.shape[1]  # Scale to image size (1280)
        if self.framerate == "images":
            flow *= 10.0  # NOTE: Scale to 1 fps approximately
        return flow.astype(np.float32)

    def _get_frame_key(self, idx: int) -> tuple[str, str]:
        """Get the frame identifier (video name, frame name) by index."""
        if len(self.scalabel_datasets) > 0:
            frames = self.scalabel_datasets[
                list(self.scalabel_datasets.keys())[0]
            ].frames
            return frames[idx].videoName, frames[idx].name
        raise ValueError("No Scalabel file has been loaded.")

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        if len(self.scalabel_datasets) > 0:
            return len(
                self.scalabel_datasets[list(self.scalabel_datasets.keys())[0]]
            )
        raise ValueError(
            "No Scalabel file has been loaded."
        )  # pragma: no cover

    def _generate_video_mapping(self) -> VideoMapping:
        """Group all dataset sample indices (int) by their video ID (str).

        Returns:
            VideoMapping: Mapping of video IDs to sample indices and frame IDs.

        Raises:
            ValueError: If no Scalabel file has been loaded.
        """
        if len(self.scalabel_datasets) > 0:
            return self.scalabel_datasets[
                list(self.scalabel_datasets.keys())[0]
            ].video_mapping
        raise ValueError("No Scalabel file has been loaded.")

    def __getitem__(self, idx: int) -> DictData:
        """Get single sample.

        Args:
            idx (int): Index of sample.

        Returns:
            DictData: sample at index in Vis4D input format.
        """
        # load camera frames
        data_dict = {}

        # metadata
        video_name, frame_name = self._get_frame_key(idx)
        data_dict[K.sample_names] = frame_name
        data_dict[K.sequence_names] = video_name
        data_dict[K.frame_ids] = frame_name.split("_")[0]

        for view in self.views_to_load:
            data_dict_view = {}

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
                if K.seg_masks in self.keys_to_load:
                    data_dict_view[K.seg_masks] = self._load(
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
            data_dict[view] = data_dict_view  # type: ignore

        return data_dict
