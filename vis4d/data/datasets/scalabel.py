"""Scalabel type dataset."""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Union

import numpy as np
import torch

from vis4d.common.distributed import broadcast
from vis4d.common.imports import SCALABEL_AVAILABLE
from vis4d.common.logging import rank_zero_info
from vis4d.common.time import Timer
from vis4d.common.typing import (
    ArgsType,
    ListAny,
    NDArrayF32,
    NDArrayI64,
    NDArrayUI8,
)
from vis4d.data.const import AxisMode
from vis4d.data.const import CommonKeys as K
from vis4d.data.datasets.util import CacheMappingMixin, DatasetFromList
from vis4d.data.io import DataBackend
from vis4d.data.typing import DictData
from vis4d.op.geometry.rotation import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
)

from .base import VideoDataset, VideoMapping
from .util import DatasetFromList, im_decode, ply_decode, print_class_histogram

if SCALABEL_AVAILABLE:
    from scalabel.label.io import load, load_label_config
    from scalabel.label.transforms import (
        box2d_to_xyxy,
        poly2ds_to_mask,
        rle_to_mask,
    )
    from scalabel.label.typing import (
        Config,
    )
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
else:
    raise ImportError("scalabel is not installed.")


def load_intrinsics(intrinsics: Intrinsics) -> NDArrayF32:
    """Transform intrinsic camera matrix according to augmentations."""
    return get_matrix_from_intrinsics(intrinsics).astype(np.float32)


def load_extrinsics(extrinsics: Extrinsics) -> NDArrayF32:
    """Transform extrinsics from Scalabel to Vis4D."""
    return get_matrix_from_extrinsics(extrinsics).astype(np.float32)


def load_image(
    url: str, backend: DataBackend, image_channel_mode: str
) -> NDArrayF32:
    """Load image tensor from url."""
    im_bytes = backend.get(url)
    image = im_decode(im_bytes, mode=image_channel_mode)
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
                label.attributes["instance_id"] = (
                    sum_previous_vids
                    + local_instance_ids[video_name].index(label.id)
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


def discard_labels_outside_set(
    dataset: list[Frame], class_set: list[str]
) -> None:
    """Discard labels outside given set of classes.

    Args:
        dataset (list[Frame]): List of frames to filter.
        class_set (list[str]): List of classes to keep.
    """
    for frame in dataset:
        remove_anns = []
        if frame.labels is not None:
            for i, ann in enumerate(frame.labels):
                if not ann.category in class_set:
                    remove_anns.append(i)
            for i in reversed(remove_anns):
                frame.labels.pop(i)


def remove_empty_samples(frames: list[Frame]) -> list[Frame]:
    """Remove empty samples."""
    new_frames = []
    for frame in frames:
        if frame.labels is None:
            continue
        labels_used = []
        for label in frame.labels:
            assert label.attributes is not None and label.category is not None
            if not check_crowd(label) and not check_ignored(label):
                labels_used.append(label)

        if len(labels_used) != 0:
            frame.labels = labels_used
            new_frames.append(frame)
    rank_zero_info(f"Filtered {len(frames) - len(new_frames)} empty frames.")
    del frames
    return new_frames


def prepare_labels(
    frames: list[Frame],
    class_list: list[str],
    global_instance_ids: bool = False,
) -> dict[str, int]:
    """Add category id and instance id to labels, return class frequencies.

    Args:
        frames (list[Frame]): List of frames.
        class_list (list[str]): List of classes.
        global_instance_ids (bool): Whether to use global instance ids.
            Defaults to False.
    """
    instance_ids: dict[str, list[str]] = defaultdict(list)
    frequencies = {cat: 0 for cat in class_list}
    for frame_id, ann in enumerate(frames):
        if ann.labels is None:  # pragma: no cover
            continue

        for label in ann.labels:
            attr: dict[str, bool | int | float | str] = {}
            if label.attributes is not None:
                attr = label.attributes

            if check_crowd(label) or check_ignored(label):
                continue

            assert label.category is not None
            frequencies[label.category] += 1
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

    return frequencies


def filter_frames_by_attributes(
    frames: list[Frame],
    attributes_to_load: Sequence[dict[str, str | float]] | None,
) -> list[Frame]:
    """Filter frames based on attributes."""
    if attributes_to_load is None:
        return frames
    filtered_frames: list[Frame] = []
    for frame in frames:
        for attribute_dict in attributes_to_load:
            if hasattr(frame, "attributes") and frame.attributes is not None:
                if all(
                    frame.attributes.get(key) == value
                    for key, value in attribute_dict.items()
                ):
                    filtered_frames.append(frame)
                    break
            else:
                raise ValueError(
                    "Attribute to load is specified but no attributes "
                    "are found in the frame."
                )
    return filtered_frames


# Not using | operator because of a bug in Python 3.9
# https://bugs.python.org/issue42233
CategoryMap = Union[dict[str, int], dict[str, dict[str, int]]]


class Scalabel(CacheMappingMixin, VideoDataset):
    """Scalabel type dataset.

    This class loads scalabel format data into Vis4D.
    """

    def __init__(
        self,
        data_root: str,
        annotation_path: str,
        keys_to_load: Sequence[str] = (K.images, K.boxes2d),
        category_map: None | CategoryMap = None,
        config_path: None | str | Config = None,
        global_instance_ids: bool = False,
        bg_as_class: bool = False,
        skip_empty_samples: bool = False,
        attributes_to_load: Sequence[dict[str, str | float]] | None = None,
        cache_as_binary: bool = False,
        cached_file_path: str | None = None,
        **kwargs: ArgsType,
    ) -> None:
        """Creates an instance of the class.

        Args:
            data_root (str): Root directory of the data.
            annotation_path (str): Path to the annotation json(s).
            keys_to_load (Sequence[str, ...], optional): Keys to load from the
                dataset. Defaults to (K.images, K.boxes2d).
            category_map (None | CategoryMap, optional): Mapping from a
                Scalabel category string to an integer index. If None, the
                standard mapping in the dataset config will be used. Defaults
                to None.
            config_path (None | str | Config, optional): Path to the dataset
                config, can be added if it is not provided together with the
                labels or should be modified. Defaults to None.
            global_instance_ids (bool): Whether to convert tracking IDs of
                annotations into dataset global IDs or stay with local,
                per-video IDs. Defaults to false.
            bg_as_class (bool): Whether to include background pixels as an
                additional class for masks.
            skip_empty_samples (bool): Whether to skip samples without
                annotations.
            attributes_to_load (Sequence[dict[str, str]]): List of attributes
                dictionaries to load. Each dictionary is a mapping from the
                attribute name to its desired value. If any of the attributes
                dictionaries is matched, the corresponding frame will be
                loaded. Defaults to None.
            cache_as_binary (bool): Whether to cache the dataset as binary.
                Default: False.
            cached_file_path (str | None): Path to a cached file. If cached
                file exist then it will load it instead of generating the data
                mapping. Default: None.
        """
        super().__init__(**kwargs)
        assert SCALABEL_AVAILABLE, "Scalabel is not installed."
        self.data_root = data_root
        self.annotation_path = annotation_path
        self.keys_to_load = keys_to_load
        self.global_instance_ids = global_instance_ids
        self.bg_as_class = bg_as_class
        self.config_path = config_path
        self.skip_empty_samples = skip_empty_samples

        self.cats_name2id: dict[str, dict[str, int]] = {}
        self.category_map = category_map

        self.attributes_to_load = attributes_to_load

        self.frames, self.cfg = self._load_mapping(
            self._generate_mapping,
            remove_empty_samples,
            cache_as_binary=cache_as_binary,
            cached_file_path=cached_file_path,
        )

        assert self.cfg is not None, (
            "No dataset configuration found. Please provide a configuration "
            "via config_path."
        )

        if self.category_map is None:
            class_list = list(
                c.name for c in get_leaf_categories(self.cfg.categories)
            )
            self.category_map = {c: i for i, c in enumerate(class_list)}
        self._setup_categories()
        self.video_mapping = self._generate_video_mapping()

    def _generate_video_mapping(self) -> VideoMapping:
        """Group all dataset sample indices (int) by their video ID (str).

        Returns:
            VideoMapping: Mapping of video IDs to sample indices and frame IDs.
        """
        video_to_indices: dict[str, list[int]] = defaultdict(list)
        video_to_frame_ids: dict[str, list[int]] = defaultdict(list)
        for idx, frame in enumerate(self.frames):  # type: ignore
            if frame.videoName is not None:
                assert (
                    frame.frameIndex is not None
                ), "found videoName but no frameIndex!"
                video_to_indices[frame.videoName].append(idx)
                video_to_frame_ids[frame.videoName].append(frame.frameIndex)

        return self._sort_video_mapping(
            {
                "video_to_indices": video_to_indices,
                "video_to_frame_ids": video_to_frame_ids,
            }
        )

    def _setup_categories(self) -> None:
        """Setup categories."""
        assert self.category_map is not None
        for target in self.keys_to_load:
            if isinstance(list(self.category_map.values())[0], int):
                self.cats_name2id[target] = self.category_map  # type: ignore
            else:
                assert (
                    target in self.category_map
                ), f"Target={target} not specified in category_mapping"
                target_map = self.category_map[target]
                assert isinstance(target_map, dict)
                self.cats_name2id[target] = target_map

    def _load_mapping(  # type: ignore
        self,
        generate_map_func: Callable[[], ScalabelData],
        filter_func: Callable[[ListAny], ListAny] = lambda x: x,
        cache_as_binary: bool = True,
        cached_file_path: str | None = None,
    ) -> tuple[DatasetFromList, Config]:
        """Load cached mapping or generate if not exists."""
        timer = Timer()
        data = self._load_mapping_data(
            generate_map_func, cache_as_binary, cached_file_path
        )
        if data is not None:
            frames, cfg = data.frames, data.config

            add_data_path(self.data_root, frames)
            rank_zero_info(f"Loading {self} takes {timer.time():.2f} seconds.")

            if self.category_map is None:
                class_list = list(
                    c.name for c in get_leaf_categories(cfg.categories)
                )
                self.category_map = {c: i for i, c in enumerate(class_list)}
            else:
                class_list = list(self.category_map.keys())

            assert len(set(class_list)) == len(
                class_list
            ), "Class names are not unique!"

            discard_labels_outside_set(frames, class_list)

            frames = filter_frames_by_attributes(
                frames, self.attributes_to_load
            )

            if self.skip_empty_samples:
                frames = filter_func(frames)

            t = Timer()
            frequencies = prepare_labels(
                frames,
                class_list,
                global_instance_ids=self.global_instance_ids,
            )
            rank_zero_info(
                f"Preprocessing {len(frames)} frames takes {t.time():.2f}"
                " seconds."
            )
            print_class_histogram(frequencies)
            frames_dataset = DatasetFromList(frames)
        else:
            frames_dataset = None
            cfg = None
        frames_dataset = broadcast(frames_dataset)
        cfg = broadcast(cfg)
        assert frames_dataset is not None
        return frames_dataset, cfg

    def _generate_mapping(self) -> ScalabelData:
        """Generate data mapping."""
        data = load(self.annotation_path)
        if self.config_path is not None:
            if isinstance(self.config_path, str):
                data.config = load_label_config(self.config_path)
            else:
                data.config = self.config_path
        return data

    def _load_inputs(self, frame: Frame) -> DictData:
        """Load inputs given a scalabel frame."""
        data: DictData = {}
        if K.images in self.keys_to_load:
            assert frame.url is not None, "url is None!"
            image = load_image(
                frame.url, self.data_backend, self.image_channel_mode
            )
            input_hw = (image.shape[1], image.shape[2])
            data[K.images] = image
            data[K.input_hw] = input_hw

            # Original image
            data[K.original_images] = image
            data[K.original_hw] = input_hw

            data[K.axis_mode] = AxisMode.OPENCV
            data[K.frame_ids] = frame.frameIndex

            data[K.sample_names] = frame.name
            data[K.sequence_names] = frame.videoName

        if K.points3d in self.keys_to_load:
            assert frame.url is not None, "url is None!"
            data[K.points3d] = load_pointcloud(frame.url, self.data_backend)

        if frame.intrinsics is not None and K.intrinsics in self.keys_to_load:
            data[K.intrinsics] = load_intrinsics(frame.intrinsics)

        if frame.extrinsics is not None and K.extrinsics in self.keys_to_load:
            data[K.extrinsics] = load_extrinsics(frame.extrinsics)
        return data

    def _add_annotations(self, frame: Frame, data: DictData) -> None:
        """Add annotations given a scalabel frame and a data dictionary."""
        labels_used, instid_map = [], {}
        if frame.labels is not None:
            for label in frame.labels:
                assert (
                    label.attributes is not None and label.category is not None
                )
                if not check_crowd(label) and not check_ignored(label):
                    labels_used.append(label)
                    if label.id not in instid_map:
                        instid_map[label.id] = int(
                            label.attributes["instance_id"]
                        )

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
                labels_used, cats_name2id, image_size
            )
            data[K.instance_masks] = instance_masks

        if K.seg_masks in self.keys_to_load:
            sem_map = self.cats_name2id[K.seg_masks]
            semantic_masks = semantic_masks_from_scalabel(
                labels_used, sem_map, image_size, self.bg_as_class
            )
            data[K.seg_masks] = semantic_masks

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

        # load annotations to input sample
        self._add_annotations(frame, data)

        return data


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
) -> NDArrayUI8:
    """Convert instance masks from scalabel format to Vis4D.

    Args:
        labels (list[Label]): list of scalabel labels.
        class_to_idx (dict[str, int]): mapping from class name to index.
        image_size (ImageSize, optional): image size. Defaults to None.

    Returns:
        NDArrayUI8: instance masks.
    """
    bitmask_list = []
    for _, label in enumerate(labels):
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
        else:
            raise ValueError("No mask found in label.")
        bitmask_list.append(bitmask)
    if len(bitmask_list) == 0:  # pragma: no cover
        return np.empty((0, 0, 0), dtype=np.uint8)
    mask_array = np.array(bitmask_list, dtype=np.uint8)
    return mask_array


def nhw_to_hwc_mask(
    masks: NDArrayUI8, class_ids: NDArrayI64, ignore_class: int = 255
) -> NDArrayUI8:
    """Convert N binary HxW masks to HxW semantic mask.

    Args:
        masks (NDArrayUI8): Masks with shape [N, H, W].
        class_ids (NDArrayI64): Class IDs with shape [N, 1].
        ignore_class (int, optional): Ignore label. Defaults to 255.

    Returns:
        NDArrayUI8: Masks with shape [H, W], where each location indicate the
            class label.
    """
    hwc_mask = np.full(masks.shape[1:], ignore_class, dtype=masks.dtype)
    for mask, cat_id in zip(masks, class_ids):
        hwc_mask[mask > 0] = cat_id
    return hwc_mask


def semantic_masks_from_scalabel(
    labels: list[Label],
    class_to_idx: dict[str, int],
    image_size: ImageSize | None = None,
    bg_as_class: bool = False,
) -> NDArrayUI8:
    """Convert masks from scalabel format to Vis4D.

    Args:
        labels (list[Label]): list of scalabel labels.
        class_to_idx (dict[str, int]): mapping from class name to index.
        image_size (ImageSize, optional): image size. Defaults to None.
        bg_as_class (bool, optional): whether to include background as a class.
            Defaults to False.

    Returns:
        NDArrayUI8: instance masks.
    """
    bitmask_list, cls_list = [], []
    if bg_as_class:
        foreground: NDArrayUI8 | None = None
    for _, label in enumerate(labels):
        if label.poly2d is None and label.rle is None:
            continue
        mask_cls = label.category
        if mask_cls in class_to_idx:
            cls_list.append(class_to_idx[mask_cls])
        else:  # pragma: no cover
            continue  # skip unknown classes
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
        else:
            raise ValueError("No mask found in label.")
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
        assert "background" in class_to_idx, (
            '"bg_as_class" requires "background" class to be '
            "in category_mapping"
        )
        cls_list.append(class_to_idx["background"])
    if len(bitmask_list) == 0:  # pragma: no cover
        return np.empty((0, 0), dtype=np.uint8)
    mask_array = np.array(bitmask_list, dtype=np.uint8)
    class_ids = np.array(cls_list, dtype=np.int64)
    return nhw_to_hwc_mask(mask_array, class_ids)


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
