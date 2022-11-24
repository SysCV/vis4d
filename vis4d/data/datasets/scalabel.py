"""Scalabel type dataset."""
from __future__ import annotations

import os
import pickle
from collections import defaultdict
from collections.abc import Callable
from typing import Union

import appdirs
import numpy as np
import torch
from torch import Tensor

from vis4d.common.imports import SCALABEL_AVAILABLE
from vis4d.common.logging import rank_zero_info
from vis4d.common.time import Timer
from vis4d.data.const import CommonKeys
from vis4d.data.datasets.util import CacheMappingMixin, DatasetFromList
from vis4d.data.io import DataBackend, FileBackend
from vis4d.data.typing import DictData

from .base import Dataset, VideoMixin
from .util import im_decode

if SCALABEL_AVAILABLE:
    from scalabel.label.io import load, load_label_config
    from scalabel.label.transforms import box2d_to_xyxy
    from scalabel.label.typing import Config
    from scalabel.label.typing import Dataset as ScalabelData
    from scalabel.label.typing import Extrinsics, Frame, Intrinsics, Label
    from scalabel.label.utils import (
        check_crowd,
        check_ignored,
        get_leaf_categories,
        get_matrix_from_extrinsics,
        get_matrix_from_intrinsics,
    )


def load_intrinsics(intrinsics: Intrinsics) -> Tensor:
    """Transform intrinsic camera matrix according to augmentations."""
    intrinsic_matrix = torch.from_numpy(
        get_matrix_from_intrinsics(intrinsics)
    ).to(torch.float32)
    return intrinsic_matrix


def load_extrinsics(extrinsics: Extrinsics) -> Tensor:
    """Transform extrinsics from Scalabel to Vis4D."""
    extrinsics_matrix = torch.from_numpy(
        get_matrix_from_extrinsics(extrinsics)
    ).to(torch.float32)
    return extrinsics_matrix


def instance_ids_to_global(
    frames: list[Frame], local_instance_ids: dict[str, list[str]]
) -> None:
    """Use local (per video) instance ids to produce global ones."""
    video_names = list(local_instance_ids.keys())
    for frame_id, ann in enumerate(frames):
        if ann.labels is None:  # pragma: no cover
            continue
        for label in ann.labels:
            assert label.attributes is not None
            if not check_crowd(label) and not check_ignored(label):
                video_name = (
                    ann.videoName
                    if ann.videoName is not None
                    else "no-video-" + str(frame_id)
                )
                sum_previous_vids = sum(
                    (
                        len(local_instance_ids[v])
                        for v in video_names[: video_names.index(video_name)]
                    )
                )
                label.attributes[
                    "instance_id"
                ] = sum_previous_vids + local_instance_ids[video_name].index(
                    label.id
                )


def add_data_path(data_root: str, frames: list[Frame]) -> None:
    """Add filepath to frame using data_root."""
    for ann in frames:
        assert ann.name is not None
        if ann.url is None:
            if ann.videoName is not None:
                ann.url = os.path.join(data_root, ann.videoName, ann.name)
            else:
                ann.url = os.path.join(data_root, ann.name)
        else:
            ann.url = os.path.join(data_root, ann.url)


def prepare_labels(
    frames: list[Frame], global_instance_ids: bool = False
) -> None:
    """Add category id and instance id to labels, return class frequencies."""
    instance_ids: dict[str, list[str]] = defaultdict(list)
    for frame_id, ann in enumerate(frames):
        if ann.labels is None:
            continue

        for label in ann.labels:
            attr: dict[str, bool | int | float | str] = {}
            if label.attributes is not None:
                attr = label.attributes

            if check_crowd(label) or check_ignored(label):
                continue

            assert label.category is not None
            video_name = (
                ann.videoName
                if ann.videoName is not None
                else "no-video-" + str(frame_id)
            )
            if label.id not in instance_ids[video_name]:
                instance_ids[video_name].append(label.id)
            attr["instance_id"] = instance_ids[video_name].index(label.id)
            label.attributes = attr

    if global_instance_ids:
        instance_ids_to_global(frames, instance_ids)


# Not using | operator because of a bug in Python 3.9
# https://bugs.python.org/issue42233
CategoryMap = Union[dict[str, int], dict[str, dict[str, int]]]


class Scalabel(Dataset, CacheMappingMixin):
    """Scalabel type dataset.

    This class loads scalabel format data into Vis4D.
    """

    def __init__(
        self,
        data_root: str,
        annotation_path: str,
        inputs_to_load: tuple[str, ...] = (CommonKeys.images,),
        targets_to_load: tuple[str, ...] = (CommonKeys.boxes2d,),
        data_backend: None | DataBackend = None,
        category_map: None | CategoryMap = None,
        config_path: None | str = None,
        global_instance_ids: bool = False,
    ) -> None:
        """Init.

        Args:
            data_root (str): Root directory of the data.
            annotation_path (str): Path to the annotation json(s).
            inputs_to_load (tuple[str, ...], optional): Input fields to load.
                Defaults to (CommonKeys.images,).
            targets_to_load (tuple[str, ...], optional): Annotation fields to
                load. Defaults to (CommonKeys.boxes2d,
                CommonKeys.boxes2d_classes).
            data_backend (None | DataBackend, optional): Data backend, if None
                then classic file backend. Defaults to None.
            category_map (None | CategoryMap, optional): Mapping from a
                Scalabel category string to an integer index. If None, the
                standard mapping in the dataset config will be used. Defaults
                to None.
            config_path (None | str, optional): Path to the dataset config, can
                be added if it is not provided together with the labels or
                should be modified. Defaults to None.
            global_instance_ids (bool): Whether to convert tracking IDs of
                annotations into dataset global IDs or stay with local,
                per-video IDs. Defaults to false.
        """
        super().__init__()
        self.data_root = data_root
        self.annotation_path = annotation_path
        self.inputs_to_load = inputs_to_load
        self.targets_to_load = targets_to_load
        self.global_instance_ids = global_instance_ids
        self.data_backend = (
            data_backend if data_backend is not None else FileBackend()
        )
        self.frames, self.cfg = self._load_mapping(self._generate_mapping)
        if config_path is not None:
            self.cfg = load_label_config(config_path)

        assert self.cfg is not None, (
            "No dataset configuration found. Please provide a configuration "
            "via config_path."
        )

        self.cats_name2id: dict[str, dict[str, int]] = {}
        if category_map is None:
            class_list = list(
                c.name for c in get_leaf_categories(self.cfg.categories)
            )
            assert len(set(class_list)) == len(
                class_list
            ), "Class names are not unique!"
            category_map = {c: i for i, c in enumerate(class_list)}
        self._setup_categories(category_map)

    def _setup_categories(self, category_map: CategoryMap) -> None:
        """Setup categories."""
        for target in self.targets_to_load:
            if isinstance(list(category_map.values())[0], int):
                self.cats_name2id[target] = category_map  # type: ignore
            else:
                assert (
                    target in category_map
                ), f"Target={target} not specified in category_mapping"
                target_map = category_map[target]
                assert isinstance(target_map, dict)
                self.cats_name2id[target] = target_map

    def _load_mapping(
        self,
        generate_map_func: Callable[[], ScalabelData],
        use_cache: bool = True,
    ) -> tuple[Dataset, Config]:
        """Load cached mapping or generate if not exists."""
        timer = Timer()
        if use_cache:
            cache_dir = os.path.join(
                appdirs.user_cache_dir(appname="vis4d"),
                "data_mapping",
                self.__class__.__name__,
            )
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, self._get_hash() + ".pkl")
            if not os.path.exists(cache_path):
                data = generate_map_func()
                with open(cache_path, "wb") as file:
                    file.write(pickle.dumps(data))
            else:
                with open(cache_path, "rb") as file:
                    data = pickle.loads(file.read())
        else:
            data = generate_map_func()

        frames, cfg = data.frames, data.config
        add_data_path(self.data_root, frames)
        prepare_labels(frames, global_instance_ids=self.global_instance_ids)
        frames = DatasetFromList(frames)
        rank_zero_info(f"Loading {self} takes {timer.time():.2f} seconds.")
        return frames, cfg

    def _generate_mapping(self) -> ScalabelData:
        """Generate data mapping."""
        return load(self.annotation_path)

    def _load_inputs(self, frame: Frame) -> DictData:
        """Load inputs given a scalabel frame."""
        data: DictData = {}
        if frame.url is not None and CommonKeys.images in self.inputs_to_load:
            im_bytes = self.data_backend.get(frame.url)
            image = im_decode(im_bytes)
            image = torch.as_tensor(
                np.ascontiguousarray(image.transpose(2, 0, 1)),
                dtype=torch.float32,
            ).unsqueeze(0)
            data[CommonKeys.images] = image
            data[CommonKeys.original_hw] = (image.shape[2], image.shape[3])
            data[CommonKeys.input_hw] = (image.shape[2], image.shape[3])
        if (
            frame.intrinsics is not None
            and CommonKeys.intrinsics in self.inputs_to_load
        ):
            data[CommonKeys.intrinsics] = load_intrinsics(frame.intrinsics)

        if (
            frame.extrinsics is not None
            and CommonKeys.extrinsics in self.inputs_to_load
        ):
            data[CommonKeys.extrinsics] = load_extrinsics(frame.extrinsics)
        return data

    def _add_annotations(self, frame: Frame, data: DictData) -> None:
        """Add annotations given a scalabel frame and a data dictionary."""
        if frame.labels is None:
            return
        labels_used, instid_map = [], {}
        for label in frame.labels:
            assert label.attributes is not None and label.category is not None
            if not check_crowd(label) and not check_ignored(label):
                labels_used.append(label)
                if label.id not in instid_map:
                    instid_map[label.id] = int(label.attributes["instance_id"])
        if not labels_used:
            return  # pragma: no cover

        if CommonKeys.masks in self.targets_to_load:
            ins_map = self.cats_name2id[CommonKeys.masks]
            instance_masks = instance_masks_from_scalabel(
                labels_used, ins_map, instid_map, frame.size
            )
            data[CommonKeys.masks] = instance_masks  # TODO sync boxes2d?

        if CommonKeys.segmentation_masks in self.targets_to_load:
            sem_map = self.cats_name2id[CommonKeys.segmentation_masks]
            semantic_masks = semantic_masks_from_scalabel(
                labels_used, sem_map, instid_map, frame.size, self.bg_as_class
            )
            data[CommonKeys.segmentation_masks] = semantic_masks

        if CommonKeys.boxes2d in self.targets_to_load:
            boxes2d, classes, track_ids = boxes2d_from_scalabel(
                labels_used, self.cats_name2id[CommonKeys.boxes2d], instid_map
            )
            data[CommonKeys.boxes2d] = boxes2d
            data[CommonKeys.boxes2d_classes] = classes
            data[CommonKeys.boxes2d_track_ids] = track_ids

    def __len__(self):
        """Length of dataset."""
        return len(self.frames)

    def __getitem__(self, index: int) -> DictData:
        """Get item from dataset at given index."""
        frame = self.frames[index]  # type: Frame
        data = self._load_inputs(frame)
        if len(self.targets_to_load) > 0:
            if len(self.cats_name2id) == 0:
                raise AttributeError(
                    "Category mapping is empty but targets_to_load is not. "
                    "Please specify a category mapping."
                )
            # load annotations to input sample
            self._add_annotations(frame, data)
        return data


class ScalabelVideo(Scalabel, VideoMixin):
    """Scalabel type dataset with video extension."""

    @property
    def video_to_indices(self) -> dict[str, list[int]]:
        """Group all dataset sample indices (int) by their video ID (str).

        Returns:
            dict[str, list[int]]: Mapping video to index.
        """
        video_to_indices: dict[str, list[int]] = defaultdict(list)
        video_to_frameidx: dict[str, list[int]] = defaultdict(list)
        for idx, frame in enumerate(self.frames):  # type: ignore
            if frame.videoName is not None:
                assert (
                    frame.frameIndex is not None
                ), "found videoName but no frameIndex!"
                video_to_frameidx[frame.videoName].append(frame.frameIndex)
                video_to_indices[frame.videoName].append(idx)

        # sort dataset indices by frame indices
        for key, idcs in video_to_indices.items():
            zip_frame_idx = sorted(zip(video_to_frameidx[key], idcs))
            video_to_indices[key] = [idx for _, idx in zip_frame_idx]
        return video_to_indices


def boxes2d_from_scalabel(
    labels: list[Label],
    class_to_idx: dict[str, int],
    label_id_to_idx: dict[str, int] | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Convert from scalabel format to Vis4D.

    NOTE: The box definition in Scalabel includes x2y2, whereas Vis4D and
    other software libraries like detectron2, mmdet do not include this,
    which is why we convert via box2d_to_xyxy.
    """
    box_list, cls_list, idx_list = [], [], []
    for i, label in enumerate(labels):
        box, box_cls, l_id = (
            label.box2d,
            label.category,
            label.id,
        )
        if box is None:
            continue
        if box_cls in class_to_idx:
            cls_list.append(class_to_idx[box_cls])

        box_list.append(box2d_to_xyxy(box))
        idx = label_id_to_idx[l_id] if label_id_to_idx is not None else i
        idx_list.append(idx)

    if len(box_list) == 0:
        return torch.empty(0, 4), torch.empty(0), torch.empty(0)

    box_tensor = torch.tensor(box_list, dtype=torch.float32)
    class_ids = torch.tensor(cls_list, dtype=torch.long)
    track_ids = torch.tensor(idx_list, dtype=torch.long)
    return box_tensor, class_ids, track_ids
