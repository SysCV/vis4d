"""Base dataset classes.

We implement a typed version of the PyTorch dataset class here. In addition, we
provide a number of Mixin classes which a dataset can inherit from to implement
additional functionality.
"""
from __future__ import annotations

from collections.abc import Sequence

from torch.utils.data import Dataset as TorchDataset

from vis4d.data.io.base import DataBackend
from vis4d.data.io.file import FileBackend
from vis4d.data.typing import DictData


class Dataset(TorchDataset[DictData]):
    """Basic pytorch dataset with defined return type."""

    # Dataset metadata.
    DESCRIPTION = ""
    HOMEPAGE = ""
    PAPER = ""
    LICENSE = ""

    # List of all keys supported by this dataset.
    KEYS: Sequence[str] = []

    def __init__(
        self,
        image_channel_mode: str = "RGB",
        data_backend: None | DataBackend = None,
    ) -> None:
        """Initialize dataset.

        Args:
            image_channel_mode (str): Image channel mode to use. Default: RGB.
            data_backend (None | DataBackend): Data backend to use.
                Default: None.
        """
        self.image_channel_mode = image_channel_mode
        self.data_backend = (
            data_backend if data_backend is not None else FileBackend()
        )

    def __len__(self) -> int:
        """Return length of dataset."""
        raise NotImplementedError

    def __getitem__(self, idx: int) -> DictData:
        """Convert single element at given index into Vis4D data format."""
        raise NotImplementedError

    def validate_keys(self, keys_to_load: Sequence[str]) -> None:
        """Validate that all keys to load are supported.

        Args:
            keys_to_load (list[str]): List of keys to load.

        Raises:
            ValueError: Raise if any key is not defined in AVAILABLE_KEYS.
        """
        for k in keys_to_load:
            if k not in self.KEYS:
                raise ValueError(f"Key '{k}' is not supported!")


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
