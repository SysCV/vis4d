"""Utility functions for datasets."""
import copy
import os
import pickle
from io import BytesIO
from typing import Any, Callable, List

import numpy as np
from PIL import Image, ImageOps
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from torch.utils.data import Dataset

from vis4d.common_to_revise.utils.imports import OPENCV_AVAILABLE
from vis4d.common_to_revise.utils.time import Timer
from vis4d.struct_to_revise import NDArrayI64, NDArrayUI8
from vis4d.struct_to_revise.structures import DictStrAny

if OPENCV_AVAILABLE:
    from cv2 import (  # pylint: disable=no-member,no-name-in-module
        COLOR_BGR2RGB,
        IMREAD_COLOR,
        cvtColor,
        imdecode,
    )


def convert_input_dir_to_dataset(input_dir: str) -> None:  # TODO revise
    """Convert a given input directory to a dataset for prediction."""
    if input_dir is not None:
        if input_dir is not None:
            if not os.path.exists(input_dir):
                raise FileNotFoundError(
                    f"Input directory does not exist: {input_dir}"
                )
        if input_dir[-1] == "/":
            input_dir = input_dir[:-1]
        dataset_name = os.path.basename(input_dir)
        dataset = ScalabelDataset(Custom(dataset_name, input_dir), False)
    return dataset


def im_decode(
    im_bytes: bytes, mode: str = "RGB", backend: str = "PIL"
) -> NDArrayUI8:
    """Decode to image (numpy array, RGB) from bytes."""
    assert mode in ["BGR", "RGB"], f"{mode} not supported for image decoding!"
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
    """Caches a mapping for fast I/O and multi-processing."""

    def _load_mapping(
        self,
        generate_map_func: Callable[[], List[DictStrAny]],
        cache_path: str,
    ) -> Dataset:
        """Load cached mapping or generate if not exists."""
        timer = Timer()
        if cache_path is not None:
            if not os.path.exists(cache_path):
                data = generate_map_func()
                with open(cache_path, "wb") as file:
                    file.write(pickle.dumps(data))
            else:
                with open(cache_path, "rb") as file:
                    data = pickle.loads(file.read())
        else:
            data = generate_map_func()

        # dataset = DatasetFromList(data)
        rank_zero_info(
            f"Loading {self.__repr__()} takes {timer.time():.2f} seconds."
        )
        return data


# reference:
# https://github.com/facebookresearch/detectron2/blob/7f8f29deae278b75625872c8a0b00b74129446ac/detectron2/data/common.py#L109
class DatasetFromList(Dataset):  # type: ignore
    """Wrap a list to a torch Dataset.

    We serialize and wrap big python objects in a torch.Dataset due to a
    memory leak when dealing with large python objects using multiple workers.
    See: https://github.com/pytorch/pytorch/issues/13246
    """

    def __init__(  # type: ignore
        self, lst: List[Any], deepcopy: bool = False, serialize: bool = True
    ):
        """Init.

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
