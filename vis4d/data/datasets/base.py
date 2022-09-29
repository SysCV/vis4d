"""Base dataset in Vis4D."""

from dataclasses import dataclass
from typing import Dict, Union

from torch import Tensor
from torch.utils.data import Dataset

DictStrArray = Dict[str, Tensor]
DictStrArrayNested = Dict[str, Union[Tensor, DictStrArray]]
DictData = Dict[str, Union[Tensor, DictStrArrayNested]]


@dataclass
class DataKeys:
    images = "images"
    boxes2d = "boxes2d"
    intrinsics = "intrinsics"
    masks = "masks"


"""DictData

This container can hold arbitrary keys of data, where data of the keys defined
in DataKeys should be in the following format:
images: Tensor of shape [1, C, H, W]
boxes2d: Tensor of shape [N, 4]

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
        """This function should group all dataset sampled indices by their
        associated video ID (str).

        Returns:
            Dict[str, int]: Mapping video to index.
        """
