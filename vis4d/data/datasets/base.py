"""Base dataset in Vis4D."""
from __future__ import annotations

from torch.utils.data import Dataset as TorchDataset

from vis4d.data.typing import DictData


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
    def video_to_indices(self) -> dict[str, list[int]]:
        """Group dataset sample indices by their associated video ID.

        The sample index is an integer while video IDs are string.

        Returns:
            dict[str, list[int]]: Mapping video to index.
        """
        raise NotImplementedError

    def get_video_indices(self, idx: int) -> list[int]:
        """Get all dataset indices in a video given a single dataset index."""
        for indices in self.video_to_indices.values():
            if idx in indices:
                return indices
        raise ValueError(f"Dataset index {idx} not found in video_to_indices!")


class MultitaskMixin:
    """Multitask dataset interface."""

    _KEYS: list[str] = []

    def validate_keys(self, keys: tuple[str, ...]) -> None:
        """Validation the keys are defined in _KEYS.

        Args:
            keys (list[str]): User input of keys to load.

        Raises:
            ValueError: Raise if any key is not defined in _KEYS.
        """
        for k in keys:
            if k not in self._KEYS:
                raise ValueError(f"Key '{k}' is not supported!")


class CategoryMapMixin:
    """Mixin for category map.

    Provides interface for filtering based on categories.
    """

    @property
    def category_to_indices(self) -> dict[str, list[int]]:
        """This function should group all dataset sample indices (int) by their
        category (str).

        Returns:
            dict[str, int]: Mapping category to index.
        """
        raise NotImplementedError

    def get_category_indices(self, idx: int) -> list[int]:
        """Get all indices of the data samples that share the same category of
        the given sample index.
        """
        for indices in self.category_to_indices.values():
            if idx in indices:
                return indices
        raise ValueError(
            f"Dataset index {idx} not found in category_to_indices!"
        )


class AttributeMapMixin:
    """Mixin for attributes map.

    Provides interface for filtering based on attributes.
    """

    @property
    def attribute_to_indices(self) -> dict[str, dict[str, list[int]]]:
        """This function should group all dataset sample indices (int) by their
        category (str).

        Returns:
            dict[str, dict[str, list[int]]]: Mapping category to index.
        """
        raise NotImplementedError


class FilteredDataset(Dataset):
    """Subset of a dataset at specified indices.

    It uses the dataset and applies filter_fn to it, which should return the
    dataset indices that are to be kept after filtering.

    Attributes:
        dataset (Dataset): The whole Dataset
        filter_fn (Dataset -> list[int]): filtering function.
    """

    def __init__(self, dataset, filter_fn) -> None:
        """Init."""
        super().__init__()
        assert isinstance(dataset, FilterMixin)  # TODO fix
        self._filtered_indices = filter_fn(dataset)

    def __len__(self) -> int:
        """Wrapper for len."""
        return len(self._filtered_indices)

    def __getitem__(self, idx):
        """Wrapper for getitem."""
        mapped_idx = self._filtered_indices[idx]
        return self.dataset[mapped_idx]
