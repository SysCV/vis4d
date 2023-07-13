"""Resample index to recover the original dataset length."""
from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset

from vis4d.common.logging import rank_zero_info

from .reference import MultiViewDataset
from .typing import DictDataOrList


class ResampleDataset(Dataset[DictDataOrList]):
    """Dataset wrapper to recover the filtered samples through resampling."""

    def __init__(self, dataset: Dataset[DictDataOrList]) -> None:
        """Creates an instance of the class."""
        super().__init__()
        self.dataset = dataset
        self.has_reference = isinstance(dataset, MultiViewDataset)
        self.valid_len = len(dataset)  # type: ignore

        # Handle the case that dataset is already wrapped.
        if hasattr(self.dataset, "dataset"):
            _dataset = self.dataset.dataset
        else:
            _dataset = self.dataset

        assert hasattr(_dataset, "original_len"), (
            "The dataset must have the attribute `original_len` to resample "
            + "index to recover the original length."
        )
        self.original_len = _dataset.original_len

        rank_zero_info(
            f"Recover {_dataset} to {self.original_len} samples by resampling "
            + "index."
        )

    def __len__(self) -> int:
        """Return the length of dataset.

        Returns:
            int: Length of dataset.
        """
        return self.original_len

    def __getitem__(self, idx: int) -> DictDataOrList:
        """Get original dataset idx according to the given index.

        Resample index to recover the original dataset length.

        Args:
            idx (int): The index of original dataset length.

        Returns:
            DictDataOrList: Data of the corresponding index.
        """
        if idx < self.valid_len:
            index = idx
        else:
            index = np.random.randint(0, self.valid_len)
        return self.dataset[index]
