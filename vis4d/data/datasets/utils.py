"""Dataset utilities"""
import copy
import os
import pickle
from typing import Any, Callable, List, Optional

import numpy as np
import torch
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from torch.utils.data import Dataset

from vis4d.common_to_revise.utils.time import Timer
from vis4d.struct_to_revise import NDArrayI64, NDArrayUI8
from vis4d.struct_to_revise.structures import DictStrAny


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
                data_list = generate_map_func()
                with open(cache_path, "wb") as file:
                    file.write(pickle.dumps(data_list))
            else:
                with open(cache_path, "rb") as file:
                    data_list = pickle.loads(file.read())
        else:
            data_list = generate_map_func()

        dataset = DatasetFromList(data_list)
        print(f"Loading {self.__repr__} takes {timer.time():.2f} seconds.")
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
