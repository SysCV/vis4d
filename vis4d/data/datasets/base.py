"""Base dataset in Vis4D."""

from typing import Dict, List, Sequence, Tuple, Union

from torch.utils.data import Dataset as TorchDataset

from ..typing import DictData


class Dataset(TorchDataset[DictData]):
    """Basic pytorch dataset with defined return type."""

    def __len__(self) -> int:
        """Return length of dataset."""
        raise NotImplementedError

    def __getitem__(self, idx: int) -> DictData:
        """Convert single element at given index into Vis4D data format."""
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
