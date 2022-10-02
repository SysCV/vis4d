"""Base dataset in Vis4D."""

from dataclasses import dataclass
from typing import Dict, Tuple, TypedDict, Union, List

from torch import Tensor
from torch.utils.data import Dataset

DictStrArray = Dict[str, Tensor]
DictStrArrayNested = Dict[str, Union[Tensor, DictStrArray]]
DictData = Dict[str, Union[Tensor, DictStrArrayNested]]


class MetaData(TypedDict):
    original_hw: Tuple[int, int]
    input_hw: Tuple[int, int]


@dataclass
class DataKeys:
    metadata = "metadata"
    images = "images"
    boxes2d = "boxes2d"
    boxes2d_classes = "boxes2d_classes"
    intrinsics = "intrinsics"
    masks = "masks"


"""DictData

This container can hold arbitrary keys of data, where data of the keys defined
in DataKeys should be in the following format:
metadata: MetaData - container for meta-information about data.
images: Tensor of shape [1, C, H, W]
boxes2d: Tensor of shape [N, 4]
boxes2d_classes: Tensor of shape [N,]

"""


class BaseDataset(Dataset):
    """Basic pytorch dataset with defined return type."""

    def __len__(self) -> int:
        """Return length of dataset."""
        raise NotImplementedError

    def __getitem__(self, idx: int) -> DictData:
        """Convert single element at given index into Vis4D data format."""
        raise NotImplementedError


class BaseVideoDataset(BaseDataset):
    """Basic pytorch video dataset."""

    @property
    def video_to_indices(self) -> Dict[str, int]:
        """This function should group all dataset sample indices (int) by their
        associated video ID (str).

        Returns:
            Dict[str, int]: Mapping video to index.
        """

class MultitaskMixin:
    """Multitask dataset interface."""
    
    _TASKS: List[str] = []
    
    def validated_tasks(self, task_to_load: List[str]) -> List[str]:
        for task in task_to_load:
            if task not in MultitaskMixin._TASKS:
                raise ValueError(f"task '{task}' is not supported!")
        return task_to_load
        
    