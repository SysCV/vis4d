"""Base dataset in Vis4D."""

from typing import Dict, List, Sequence, Tuple, Union
from unittest.loader import VALID_MODULE_NAME

from torch.utils.data import Dataset as TorchDataset

from vis4d.common import DictData, MultiSensorData
from vis4d.common.typing import COMMON_KEYS


class Dataset(TorchDataset[DictData]):
    """Basic pytorch dataset with defined return type."""

    def __len__(self) -> int:
        """Return length of dataset."""
        raise NotImplementedError

    def __getitem__(self, idx: int) -> DictData:
        """Convert single element at given index into Vis4D data format."""
        raise NotImplementedError


class MultiSensorDataset(TorchDataset[MultiSensorData]):
    """Basic Multi-Sensor Dataset."""

    def __len__(self) -> int:
        """Return length of dataset."""
        raise NotImplementedError

    def __getitem__(self, idx: int) -> MultiSensorData:
        """Prepare and return multi sensor input data given an index."""
        raise NotImplementedError


class VideoMixin:
    """Mixin for video datasets.

    Provides interface for video based data and reference view samplers.
    """

    @property
    def video_to_indices(self) -> Dict[str, List[int]]:
        """This function should group all dataset sample indices (int) by their
        associated video ID (str).

        Returns:
            Dict[str, int]: Mapping video to index.
        """
        raise NotImplementedError

    def get_video_indices(self, idx: int) -> List[int]:
        """Get all dataset indices in a video given a single dataset index."""
        for indices in self.video_to_indices.values():
            if idx in indices:
                return indices
        raise ValueError(f"Dataset index {idx} not found in video_to_indices!")


class MultitaskMixin:
    """Multitask dataset interface."""

    _KEYS: List[str] = []

    def validate_keys(self, keys: Tuple[str, ...]) -> None:
        """Validation the keys are defined in _KEYS.

        Args:
            keys (List[str]): User input of keys to load.

        Raises:
            ValueError: Raise if any key is not defined in _KEYS.
        """
        for k in keys:
            if k not in self._KEYS:
                raise ValueError(f"Key '{k}' is not supported!")


class CategoryMapMixin:
    def __init__(
        self,
        categories: List[int],
        category_fields: List[str] = [
            COMMON_KEYS.boxes2d_classes,
            COMMON_KEYS.boxes3d_classes,
        ],
    ) -> None:
        self.categories = categories
        self.category_fields = category_fields


class FilteredDataset(Dataset):
    """Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        filter_fn (Dataset -> List[int]): filtering function.
    """

    def __init__(self, dataset, filter_fn) -> None:
        super().__init__()
        assert isinstance(dataset, FilterMixin)
        self._filtered_indices = filter_fn(dataset)

    def __len__(self) -> int:
        return len(self._filtered_indices)

    def __getitem__(self, idx):
        mapped_idx = self._filtered_indices[idx]
        return self.dataset[mapped_idx]
