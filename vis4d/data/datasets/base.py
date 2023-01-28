"""Base dataset classes.

We implement a typed version of the PyTorch dataset class here. In addition, we
provide a number of Mixin classes which a dataset can inherit from to implement
additional functionality.
"""
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
        """Group all dataset sample indices (int) by their category (str).

        Returns:
            dict[str, int]: Mapping category to index.
        """
        raise NotImplementedError

    def get_category_indices(self, idx: int) -> list[int]:
        """Get all indices that share the same category of the given index.

        Indices refer to the index of the data samples within the dataset.
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
        """Groups all dataset sample indices (int) by their category (str).

        Returns:
            dict[str, dict[str, list[int]]]: Mapping category to index.
        """
        raise NotImplementedError
