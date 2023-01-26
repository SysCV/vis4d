"""Utility functions for datasets."""
from __future__ import annotations

import copy
import hashlib
import os
import pickle
from collections.abc import Callable
from io import BytesIO
from typing import Any

import appdirs
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from vis4d.common import DictStrAny, NDArrayI64, NDArrayUI8
from vis4d.common.imports import OPENCV_AVAILABLE
from vis4d.common.logging import rank_zero_info
from vis4d.common.time import Timer

if OPENCV_AVAILABLE:
    from cv2 import (  # pylint: disable=no-member,no-name-in-module
        COLOR_BGR2RGB,
        IMREAD_COLOR,
        cvtColor,
        imdecode,
    )


def im_decode(
    im_bytes: bytes, mode: str = "RGB", backend: str = "PIL"
) -> NDArrayUI8:
    """Decode to image (numpy array, RGB) from bytes."""
    assert mode in {"BGR", "RGB"}, f"{mode} not supported for image decoding!"
    if backend == "PIL":
        pil_img = Image.open(BytesIO(bytearray(im_bytes)))
        pil_img = ImageOps.exif_transpose(pil_img)
        if pil_img.mode == "L":  # pragma: no cover
            # convert grayscale image to RGB
            pil_img = pil_img.convert("RGB")
        if mode == "BGR":  # pragma: no cover
            img: NDArrayUI8 = np.array(pil_img)[..., [2, 1, 0]]
        elif mode == "RGB":
            img = np.array(pil_img)
    elif backend == "cv2":  # pragma: no cover
        if not OPENCV_AVAILABLE:
            raise ImportError(
                "Please install opencv-python to use cv2 backend!"
            )
        img_np: NDArrayUI8 = np.frombuffer(im_bytes, np.uint8)
        img = imdecode(img_np, IMREAD_COLOR)
        if mode == "RGB":
            cvtColor(img, COLOR_BGR2RGB, img)
    else:
        raise NotImplementedError(f"Image backend {backend} not known!")
    return img


class CacheMappingMixin:
    """Caches a mapping for fast I/O and multi-processing.

    This class provides functionality for caching a mapping from dataset index
    requested by a call on __getitem__ to a dictionary that holds relevant
    information for loading the sample in question from the disk.
    Caching the mapping reduces startup time by loading the mapping instead of
    re-computing it at every startup.

    NOTE: The mapping will detect changes in the dataset by inspecting the
    string representation (__repr__) of your dataset. Make sure your __repr__
    implementation contains all parameters relevant to your mapping, so that
    the mapping will get updated once one of those parameters is changed.
    Conversely, make sure all non-relevant information is excluded from the
    string representation, so that the mapping can be loaded and re-used.
    """

    def _load_mapping_data(
        self,
        generate_map_func: Callable[[], list[DictStrAny]],
        use_cache: bool = True,
    ) -> list[DictStrAny]:
        """Load possibly cached mapping via generate_map_func."""
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
        return data

    def _load_mapping(
        self,
        generate_map_func: Callable[[], list[DictStrAny]],
        use_cache: bool = True,
    ) -> Dataset[DictStrAny]:
        """Load cached mapping or generate if not exists."""
        timer = Timer()
        data = self._load_mapping_data(generate_map_func, use_cache)
        dataset = DatasetFromList(data)
        rank_zero_info(
            f"Loading {str(self.__repr__)} takes {timer.time():.2f} seconds."
        )
        return dataset

    def _get_hash(self, length: int = 16) -> str:
        """Get hash of current dataset instance."""
        hasher = hashlib.sha256()
        hasher.update(str(self.__repr__).encode("utf8"))
        hash_value = hasher.hexdigest()[:length]
        return hash_value


# reference:
# https://github.com/facebookresearch/detectron2/blob/7f8f29deae278b75625872c8a0b00b74129446ac/detectron2/data/common.py#L109
class DatasetFromList(Dataset):  # type: ignore
    """Wrap a list to a torch Dataset.

    We serialize and wrap big python objects in a torch.Dataset due to a
    memory leak when dealing with large python objects using multiple workers.
    See: https://github.com/pytorch/pytorch/issues/13246
    """

    def __init__(  # type: ignore
        self, lst: list[Any], deepcopy: bool = False, serialize: bool = True
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
