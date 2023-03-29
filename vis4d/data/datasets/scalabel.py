"""Scalabel type dataset."""
from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Union

import numpy as np
import torch

from vis4d.common.imports import SCALABEL_AVAILABLE
from vis4d.common.logging import rank_zero_info
from vis4d.common.time import Timer
from vis4d.common.typing import (
    DictStrAny,
    ListAny,
    NDArrayF32,
    NDArrayI64,
    NDArrayUI8,
)
from vis4d.data.const import AxisMode
from vis4d.data.const import CommonKeys as K
from vis4d.data.datasets.util import CacheMappingMixin, DatasetFromList
from vis4d.data.io import DataBackend, FileBackend
from vis4d.data.typing import DictData
from vis4d.op.geometry.rotation import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
)

from .base import Dataset, VideoMixin
from .util import im_decode, ply_decode

if SCALABEL_AVAILABLE:
    from scalabel.label.io import load, load_label_config
    from scalabel.label.transforms import (
        box2d_to_xyxy,
        poly2ds_to_mask,
        rle_to_mask,
    )
    from scalabel.label.typing import Config
    from scalabel.label.typing import Dataset as ScalabelData
    from scalabel.label.typing import (
        Extrinsics,
        Frame,
        ImageSize,
        Intrinsics,
        Label,
    )
    from scalabel.label.utils import (
        check_crowd,
        check_ignored,
        get_leaf_categories,
        get_matrix_from_extrinsics,
        get_matrix_from_intrinsics,
    )


def load_intrinsics(intrinsics: Intrinsics) -> NDArrayF32:
    """Transform intrinsic camera matrix according to augmentations."""
    return get_matrix_from_intrinsics(intrinsics).astype(np.float32)


def load_extrinsics(extrinsics: Extrinsics) -> NDArrayF32:
    """Transform extrinsics from Scalabel to Vis4D."""
    return get_matrix_from_extrinsics(extrinsics).astype(np.float32)


def load_image(url: str, backend: DataBackend) -> NDArrayF32:
    """Load image tensor from url."""
    im_bytes = backend.get(url)
    image = im_decode(im_bytes)
    return np.ascontiguousarray(image, dtype=np.float32)[None]


def load_pointcloud(url: str, backend: DataBackend) -> NDArrayF32:
    """Load pointcloud tensor from url."""
    assert url.endswith(".ply"), "Only PLY files are supported now."
    ply_bytes = backend.get(url)
    pointcloud = ply_decode(ply_bytes)
    return pointcloud.astype(np.float32)


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
        keys_to_load: Sequence[str] = (
            K.images,
            K.boxes2d,
        ),
        data_backend: None | DataBackend = None,
        category_map: None | CategoryMap = None,
        config_path: None | str = None,
        global_instance_ids: bool = False,
        bg_as_class: bool = False,
    ) -> None:
        """Creates an instance of the class.

        Args:
            data_root (str): Root directory of the data.
            annotation_path (str): Path to the annotation json(s).
            keys_to_load (Sequence[str, ...], optional): Keys to load from the
                dataset. Defaults to (K.images, K.boxes2d).
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
            bg_as_class (bool): Whether to include background pixels as an
                additional class for masks.
        """
        super().__init__()
        self.data_root = data_root
        self.annotation_path = annotation_path
        self.keys_to_load = keys_to_load
        self.global_instance_ids = global_instance_ids
        self.bg_as_class = bg_as_class
        self.data_backend = (
            data_backend if data_backend is not None else FileBackend()
        )
        self.config_path = config_path
        self.frames, self.cfg = self._load_mapping(
            self._generate_mapping  # type: ignore
        )

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
        for target in self.keys_to_load:
            if isinstance(list(category_map.values())[0], int):
                self.cats_name2id[target] = category_map  # type: ignore
            else:
                assert (
                    target in category_map
                ), f"Target={target} not specified in category_mapping"
                target_map = category_map[target]
                assert isinstance(target_map, dict)
                self.cats_name2id[target] = target_map

    def _load_mapping(  # type: ignore
        self,
        generate_map_func: Callable[[], list[DictStrAny]],
        use_cache: bool = True,
    ) -> tuple[ListAny, Config]:
        """Load cached mapping or generate if not exists."""
        timer = Timer()
        data = self._load_mapping_data(generate_map_func, use_cache)
        frames, cfg = data.frames, data.config  # type: ignore
        add_data_path(self.data_root, frames)
        prepare_labels(frames, global_instance_ids=self.global_instance_ids)
        frames = DatasetFromList(frames)
        rank_zero_info(f"Loading {self} takes {timer.time():.2f} seconds.")
        return frames, cfg

    def _generate_mapping(self) -> ScalabelData:
        """Generate data mapping."""
        data = load(self.annotation_path)
        data.frames = sorted(data.frames, key=lambda x: x.name)
        if self.config_path is not None:
            data.config = load_label_config(self.config_path)
        return data

    def _load_inputs(self, frame: Frame) -> DictData:
        """Load inputs given a scalabel frame."""
        data: DictData = {}
        if frame.url is not None and K.images in self.keys_to_load:
            image = load_image(frame.url, self.data_backend)
            input_hw = (image.shape[1], image.shape[2])
            data[K.images] = image
            data[K.original_hw] = input_hw
            data[K.input_hw] = input_hw
            data[K.axis_mode] = AxisMode.OPENCV
            data[K.frame_ids] = frame.frameIndex
            # TODO how to properly integrate such metadata?
            data["name"] = frame.name
            data["videoName"] = frame.videoName

        if frame.url is not None and K.points3d in self.keys_to_load:
            data[K.points3d] = load_pointcloud(frame.url, self.data_backend)

        if frame.intrinsics is not None and K.intrinsics in self.keys_to_load:
            data[K.intrinsics] = load_intrinsics(frame.intrinsics)

        if frame.extrinsics is not None and K.extrinsics in self.keys_to_load:
            data[K.extrinsics] = load_extrinsics(frame.extrinsics)
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
        # if not labels_used:
        #     return  # pragma: no cover

        image_size = (
            ImageSize(height=data[K.input_hw][0], width=data[K.input_hw][1])
            if K.input_hw in data
            else frame.size
        )

        if K.boxes2d in self.keys_to_load:
            cats_name2id = self.cats_name2id[K.boxes2d]
            boxes2d, classes, track_ids = boxes2d_from_scalabel(
                labels_used, cats_name2id, instid_map
            )
            data[K.boxes2d] = boxes2d
            data[K.boxes2d_classes] = classes
            data[K.boxes2d_track_ids] = track_ids

        if K.instance_masks in self.keys_to_load:
            # NOTE: instance masks' mapping is consistent with boxes2d
            cats_name2id = self.cats_name2id[K.instance_masks]
            instance_masks = instance_masks_from_scalabel(
                labels_used,
                cats_name2id,
                image_size=image_size,
                bg_as_class=self.bg_as_class,
            )
            data[K.instance_masks] = instance_masks

        if K.segmentation_masks in self.keys_to_load:
            sem_map = self.cats_name2id[K.segmentation_masks]
            semantic_masks = semantic_masks_from_scalabel(
                labels_used, sem_map, instid_map, image_size, self.bg_as_class
            )
            data[K.segmentation_masks] = semantic_masks

        if K.boxes3d in self.keys_to_load:
            boxes3d, classes, track_ids = boxes3d_from_scalabel(
                labels_used, self.cats_name2id[K.boxes3d], instid_map
            )
            data[K.boxes3d] = boxes3d
            data[K.boxes3d_classes] = classes
            data[K.boxes3d_track_ids] = track_ids

    def __len__(self) -> int:
        """Length of dataset."""
        return len(self.frames)

    def __getitem__(self, index: int) -> DictData:
        """Get item from dataset at given index."""
        frame = self.frames[index]
        data = self._load_inputs(frame)
        if len(self.keys_to_load) > 0:
            if len(self.cats_name2id) == 0:
                raise AttributeError(
                    "Category mapping is empty but keys_to_load is not. "
                    "Please specify a category mapping."
                )  # pragma: no cover
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
        for idx, frame in enumerate(self.frames):
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


def boxes3d_from_scalabel(
    labels: list[Label],
    class_to_idx: dict[str, int],
    label_id_to_idx: dict[str, int] | None = None,
) -> tuple[NDArrayF32, NDArrayI64, NDArrayI64]:
    """Convert 3D bounding boxes from scalabel format to Vis4D."""
    box_list, cls_list, idx_list = [], [], []
    for i, label in enumerate(labels):
        box, box_cls, l_id = label.box3d, label.category, label.id
        if box is None:
            continue
        if box_cls in class_to_idx:
            cls_list.append(class_to_idx[box_cls])
        else:
            continue

        quaternion = (
            matrix_to_quaternion(
                euler_angles_to_matrix(torch.tensor([box.orientation]))
            )[0]
            .numpy()
            .tolist()
        )
        box_list.append([*box.location, *box.dimension, *quaternion])
        idx = label_id_to_idx[l_id] if label_id_to_idx is not None else i
        idx_list.append(idx)

    if len(box_list) == 0:
        return (
            np.empty((0, 10), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )
    box_tensor = np.array(box_list, dtype=np.float32)
    class_ids = np.array(cls_list, dtype=np.int64)
    track_ids = np.array(idx_list, dtype=np.int64)
    return box_tensor, class_ids, track_ids


def boxes2d_from_scalabel(
    labels: list[Label],
    class_to_idx: dict[str, int],
    label_id_to_idx: dict[str, int] | None = None,
) -> tuple[NDArrayF32, NDArrayI64, NDArrayI64]:
    """Convert from scalabel format to Vis4D.

    NOTE: The box definition in Scalabel includes x2y2 in the box area, whereas
    Vis4D and other software libraries like detectron2 and mmdet do not include
    this, which is why we convert via box2d_to_xyxy.

    Args:
        labels (list[Label]): list of scalabel labels.
        class_to_idx (dict[str, int]): mapping from class name to index.
        label_id_to_idx (dict[str, int] | None, optional): mapping from label
            id to index. Defaults to None.

    Returns:
        tuple[NDArrayF32, NDArrayI64, NDArrayI64]: boxes, classes, track_ids
    """
    box_list, cls_list, idx_list = [], [], []
    for i, label in enumerate(labels):
        box, box_cls, l_id = label.box2d, label.category, label.id
        if box is None:
            continue
        if box_cls in class_to_idx:
            cls_list.append(class_to_idx[box_cls])
        else:
            continue

        box_list.append(box2d_to_xyxy(box))
        idx = label_id_to_idx[l_id] if label_id_to_idx is not None else i
        idx_list.append(idx)

    if len(box_list) == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )

    box_tensor = np.array(box_list, dtype=np.float32)
    class_ids = np.array(cls_list, dtype=np.int64)
    track_ids = np.array(idx_list, dtype=np.int64)
    return box_tensor, class_ids, track_ids


def instance_masks_from_scalabel(
    labels: list[Label],
    class_to_idx: dict[str, int],
    image_size: ImageSize | None = None,
    bg_as_class: bool = False,
) -> NDArrayUI8:
    """Convert from scalabel format to Vis4D.

    Args:
        labels (list[Label]): list of scalabel labels.
        class_to_idx (dict[str, int]): mapping from class name to index.
        image_size (ImageSize, optional): image size. Defaults to None.
        bg_as_class (bool, optional): whether to include background as a class.
            Defaults to False.

    Returns:
        NDArrayUI8: instance masks.
    """
    bitmask_list = []
    if bg_as_class:
        foreground: NDArrayUI8 | None = None
    for label in labels:
        if label.category not in class_to_idx:  # pragma: no cover
            continue  # skip unknown classes
        if label.poly2d is None and label.rle is None:
            continue
        if label.rle is not None:
            bitmask = rle_to_mask(label.rle)
        elif label.poly2d is not None:
            assert (
                image_size is not None
            ), "image size must be specified for masks with polygons!"
            bitmask_raw = poly2ds_to_mask(image_size, label.poly2d)
            bitmask: NDArrayUI8 = (bitmask_raw > 0).astype(  # type: ignore
                bitmask_raw.dtype
            )
        bitmask_list.append(bitmask)
        if bg_as_class:
            foreground = (
                bitmask
                if foreground is None
                else np.logical_or(foreground, bitmask)
            )
    if bg_as_class:
        if foreground is None:  # pragma: no cover
            assert image_size is not None
            foreground = np.zeros(
                (image_size.height, image_size.width), dtype=np.uint8
            )
        bitmask_list.append(np.logical_not(foreground))
    if len(bitmask_list) == 0:  # pragma: no cover
        return np.empty((0, 0, 0), dtype=np.uint8)
    return np.array(bitmask_list, dtype=np.uint8)


def semantic_masks_from_scalabel(
    labels: list[Label],
    class_to_idx: dict[str, int],
    label_id_to_idx: dict[str, int] | None = None,
    frame_size: ImageSize | None = None,
    bg_as_class: bool = False,
) -> NDArrayUI8:
    """Convert from scalabel format to Vis4D."""
    raise NotImplementedError  # pragma: no cover
