"""Utility functions for datasets."""
from __future__ import annotations

import copy

import os
import pickle
from collections.abc import Callable, Sequence
from io import BytesIO
from typing import Any

import numpy as np
import plyfile
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from vis4d.common.distributed import rank_zero_only, broadcast
from vis4d.common.imports import OPENCV_AVAILABLE
from vis4d.common.logging import rank_zero_info
from vis4d.common.time import Timer
from vis4d.common.typing import (
    ListAny,
    DictStrAny,
    NDArrayFloat,
    NDArrayI64,
    NDArrayUI8,
)
from ..typing import DictData

if OPENCV_AVAILABLE:
    from cv2 import (  # pylint: disable=no-member,no-name-in-module
        COLOR_BGR2RGB,
        IMREAD_COLOR,
        IMREAD_GRAYSCALE,
        cvtColor,
        imdecode,
    )


def im_decode(
    im_bytes: bytes, mode: str = "RGB", backend: str = "PIL"
) -> NDArrayUI8:
    """Decode to image (numpy array, RGB) from bytes."""
    assert mode in {
        "BGR",
        "RGB",
        "L",
    }, f"{mode} not supported for image decoding!"
    if backend == "PIL":
        pil_img = Image.open(BytesIO(bytearray(im_bytes)))
        pil_img = ImageOps.exif_transpose(pil_img)
        if pil_img.mode == "L":  # pragma: no cover
            if mode == "L":
                img: NDArrayUI8 = np.array(pil_img)[..., None]
            else:
                # convert grayscale image to RGB
                pil_img = pil_img.convert("RGB")
        elif mode == "L":  # pragma: no cover
            raise ValueError("Cannot convert colorful image to grayscale!")
        if mode == "BGR":  # pragma: no cover
            img = np.array(pil_img)[..., [2, 1, 0]]
        elif mode == "RGB":
            img = np.array(pil_img)
    elif backend == "cv2":  # pragma: no cover
        if not OPENCV_AVAILABLE:
            raise ImportError(
                "Please install opencv-python to use cv2 backend!"
            )
        img_np: NDArrayUI8 = np.frombuffer(im_bytes, np.uint8)
        img = imdecode(
            img_np, IMREAD_GRAYSCALE if mode == "L" else IMREAD_COLOR
        )
        if mode == "RGB":
            cvtColor(img, COLOR_BGR2RGB, img)
    else:
        raise NotImplementedError(f"Image backend {backend} not known!")
    return img


def ply_decode(ply_bytes: bytes, mode: str = "XYZI") -> NDArrayFloat:
    """Decode to point clouds (numpy array) from bytes.

    Args:
        ply_bytes (bytes): The bytes of the ply file.
        mode (str, optional): The point format of the ply file. If "XYZI", the
            intensity channel will be included, otherwise only the XYZ
            coordinates. Defaults to "XYZI".
    """
    assert mode in {
        "XYZ",
        "XYZI",
    }, f"{mode} not supported for points decoding!"

    plydata = plyfile.PlyData.read(BytesIO(bytearray(ply_bytes)))
    num_points = plydata["vertex"].count
    num_channels = 3 if mode == "XYZ" else 4
    points = np.zeros((num_points, num_channels), dtype=np.float32)

    points[:, 0] = plydata["vertex"].data["x"]
    points[:, 1] = plydata["vertex"].data["y"]
    points[:, 2] = plydata["vertex"].data["z"]
    if mode == "XYZI":
        points[:, 3] = plydata["vertex"].data["intensity"]
    return points


def npy_decode(npy_bytes: bytes, key: str | None = None) -> NDArrayFloat:
    """Decode to numpy array from npy/npz file bytes."""
    data = np.load(BytesIO(bytearray(npy_bytes)))
    if key is not None:
        data = data[key]
    return data


def filter_by_keys(
    data_dict: DictData, keys_to_keep: Sequence[str]
) -> DictData:
    """Filter a dictionary by keys.

    Args:
        data_dict (DictData): The dictionary to filter.
        keys_to_keep (list[str]): The keys to keep.

    Returns:
        DictData: The filtered dictionary.
    """
    return {key: data_dict[key] for key in keys_to_keep if key in data_dict}


def get_used_data_groups(
    data_groups: dict[str, list[str]], keys: list[str]
) -> list[str]:
    """Get the data groups that are used by the given keys.

    Args:
        data_groups (dict[str, list[str]]): The data groups.
        keys (list[str]): The keys to check.

    Returns:
        list[str]: The used data groups.
    """
    used_groups = []
    for group_name, group_keys in data_groups.items():
        if not group_keys:
            continue
        if any(key in keys for key in group_keys):
            used_groups.append(group_name)
    return used_groups


# TODO: Add support for generate different timestamp cached mapping files,
# and let the user whehter generate every time or choose which one to use.
# By default, use the latest one. Add the SHA hash of the annotations file
class CacheMappingMixin:
    """Caches a mapping for fast I/O and multi-processing.

    This class provides functionality for caching a mapping from dataset index
    requested by a call on __getitem__ to a dictionary that holds relevant
    information for loading the sample in question from the disk.
    Caching the mapping reduces startup time by loading the mapping instead of
    re-computing it at every startup.

    NOTE: Make sure your annotations file is up-to-date and that the strings
    used in the mapping are unique. Otherwise, the mapping will be wrong and
    you will get wrong samples.
    """

    @rank_zero_only
    def _load_mapping_data(
        self,
        generate_map_func: Callable[[], list[DictStrAny]],
        cache_as_binary: bool,
        cached_file_path: str | None,
    ) -> DatasetFromList:
        """Load possibly cached mapping via generate_map_func.

        Args:
            generate_map_func (Callable[[], list[DictStrAny]]): The function
                that generates the mapping.
            cache_as_binary (bool): Whether to cache the mapping as binary.
            cached_file_path (str | None): The path to the cached mapping file.
        """
        if cache_as_binary:
            assert (
                cached_file_path is not None
            ), "cached_file_path must be set if cache_as_binary is True!"
            if not os.path.exists(cached_file_path):
                rank_zero_info(
                    f"Didn't find {cached_file_path} so generating it..."
                )
                data = generate_map_func()
                with open(cached_file_path, "wb") as file:
                    file.write(pickle.dumps(data))
            else:
                rank_zero_info(f"Found {cached_file_path} so loading it...")
                with open(cached_file_path, "rb") as file:
                    data = pickle.loads(file.read())
        else:
            rank_zero_info(f"Generating {self} data mapping...")
            data = generate_map_func()
        return data

    def _load_mapping(
        self,
        generate_map_func: Callable[[], list[DictStrAny]],
        filter_func: Callable[[ListAny], ListAny] = lambda x: x,
        cache_as_binary: bool = False,
        cached_file_path: str | None = None,
    ) -> DatasetFromList:
        """Load cached mapping or generate if not exists.

        Args:
            generate_map_func (Callable[[], list[DictStrAny]]): The function
                that generates the mapping.
            filter_func (Callable[[ListAny], ListAny], optional): The function
                that filters the mapping. Defaults to lambda x: x.
            cache_as_binary (bool, optional): Whether to cache the mapping as
                binary. Defaults to True.
            cached_file_path (str | None, optional): The path to the cached
                mapping file. Defaults to None.
        """
        timer = Timer()
        dataset = self._load_mapping_data(
            generate_map_func, cache_as_binary, cached_file_path
        )
        if dataset is not None:
            dataset = filter_func(dataset)
            dataset = DatasetFromList(dataset)
        dataset = broadcast(dataset)
        rank_zero_info(f"Loading {self} takes {timer.time():.2f} seconds.")
        return dataset


# reference:
# https://github.com/facebookresearch/detectron2/blob/7f8f29deae278b75625872c8a0b00b74129446ac/detectron2/data/common.py#L109
class DatasetFromList(Dataset):  # type: ignore
    """Wrap a list to a torch Dataset.

    We serialize and wrap big python objects in a torch.Dataset due to a
    memory leak when dealing with large python objects using multiple workers.
    See: https://github.com/pytorch/pytorch/issues/13246
    """

    def __init__(  # type: ignore
        self, lst: ListAny, deepcopy: bool = False, serialize: bool = True
    ):
        """Creates an instance of the class.

        Args:
            lst: a list which contains elements to produce.
            deepcopy: whether to deepcopy the element when producing it, s.t.
            the result can be modified in place without affecting the source
            in the list.
            serialize: whether to hold memory using serialized objects. When
            enabled, data loader workers can use shared RAM from master
            process instead of making a copy.
        """
        self._copy = deepcopy
        self._serialize = serialize

        def _serialize(data: Any) -> NDArrayUI8:  # type: ignore
            """Serialize python object to numpy array."""
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        if self._serialize:
            self._lst = [_serialize(x) for x in lst]
            self._addr: NDArrayI64 = np.asarray(
                [len(x) for x in self._lst], dtype=np.int64
            )
            self._addr = np.cumsum(self._addr)
            self._lst = np.concatenate(self._lst)  # type: ignore
        else:
            self._lst = lst  # pragma: no cover

    def __len__(self) -> int:
        """Return len of list."""
        if self._serialize:
            return len(self._addr)
        return len(self._lst)  # pragma: no cover

    def __getitem__(self, idx: int) -> Any:  # type: ignore
        """Return item of list at idx."""
        if self._serialize:
            start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
            end_addr = self._addr[idx].item()
            bytes_ = memoryview(self._lst[start_addr:end_addr])  # type: ignore
            return pickle.loads(bytes_)
        if self._copy:  # pragma: no cover
            return copy.deepcopy(self._lst[idx])

        return self._lst[idx]  # pragma: no cover
