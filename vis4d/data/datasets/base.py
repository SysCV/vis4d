"""Base dataset in Vis4D."""

import sys
from dataclasses import dataclass
from tkinter import image_names
from typing import Dict, List, Sequence, Tuple, Union

from vis4d.struct_to_revise.structures import DictStrAny

if sys.version_info >= (3, 8):
    from typing import TypedDict  # pylint: disable=no-name-in-module
else:
    from typing_extensions import TypedDict

from torch import Tensor
from torch.utils.data import Dataset as TorchDataset

_DictStrArray = Dict[str, Tensor]
_DictStrArrayNested = Dict[str, Union[Tensor, _DictStrArray]]
DictData = Dict[str, Union[Tensor, _DictStrArrayNested]]




@dataclass
class DataKeys:
    """DataKeys defines the supported keys for DictData.

    This container can hold arbitrary keys of data, where data of the keys defined
    in DataKeys should be in the following format:
    metadata: MetaData - container for meta-information about data.

    original_hw: Tuple[int, int]
    input_hw: Tuple[int, int]
    transform_params: DictStrAny
    batch_transform_params: DictStrAny
    images: Tensor of shape [1, C, H, W]
    boxes2d: Tensor of shape [N, 4]
    boxes2d_classes: Tensor of shape [N,]
    masks: Tensor of shape [N, H, W]
    """

    original_hw = "original_hw"
    input_hw = "input_hw"
    transform_params = "transform_params"
    batch_transform_params = "batch_transform_params"
    image = "image"
    boxes2d = "boxes2d"
    boxes2d_classes = "boxes2d_classes"
    intrinsics = "intrinsics"
    extrinsics = "extrinsiscs"
    timestamp = "timestamp"
    masks = "masks"
    segmentation_mask = "segmentation_mask"
    points3d = "points3d"
    colors3d = "colors3d"
    semantics3d = "semantics3d"
    instances3d = "instances3d"
    boxes3d = "boxes3d"
    boxes3d_classes = "boxes3d_classes"


class Dataset(TorchDataset[DictData]):
    """Basic pytorch dataset with defined return type."""

    def __len__(self) -> int:
        """Return length of dataset."""
        raise NotImplementedError

    def __getitem__(self, idx: int) -> DictData:
        """Convert single element at given index into Vis4D data format."""
        raise NotImplementedError


#class MultiViewDataset(TorchDataset[MultiViewDataDict]): TODO


class VideoDataset(Dataset):
    """Basic pytorch video dataset."""

    @property
    def video_to_indices(self) -> Dict[str, List[int]]:
        """This function should group all dataset sample indices (int) by their
        associated video ID (str).

        Returns:
            Dict[str, int]: Mapping video to index.
        """


class NuScenes(Dataset, MultitaskMixin):
    _KEYS = ["FRONT_CAM"]

    -> front cam iamge


datasets = [
    NuScenes(keys = ["FRONT_CAM", "BACK_CAM"]),
    NuScenes(keys = "BACK_CAM"),
    ...
]

faster_rcnn



class MultitaskMixin:
    """Multitask dataset interface."""

    _KEYS: List[str] = []

    def validate_keys(self, keys_to_load: List[str]) -> bool:
        for k in keys_to_load:
            if k not in self._KEYS:
                raise ValueError(f"Key '{k}' is not supported!")
        return True


class Subset(Dataset):
    """Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset: Dataset, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx: Union[int, List[int]]) -> DictData:
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __len__(self) -> int:
        return len(self.indices)
