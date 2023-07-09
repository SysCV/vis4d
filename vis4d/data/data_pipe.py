"""DataPipe wraps datasets to share the prepossessing pipeline."""
from __future__ import annotations

import random
from collections.abc import Callable, Iterable

from torch.utils.data import ConcatDataset, Dataset

from .reference import MultiViewDataset
from .typing import DictData, DictDataOrList


class DataPipe(ConcatDataset[DictDataOrList]):
    """DataPipe class.

    This class wraps one or multiple instances of a PyTorch Dataset so that the
    preprocessing steps can be shared across those datasets. Composes dataset
    and the preprocessing pipeline.
    """

    def __init__(
        self,
        datasets: Dataset[DictDataOrList] | Iterable[Dataset[DictDataOrList]],
        preprocess_fn: Callable[
            [list[DictData]], list[DictData]
        ] = lambda x: x,
    ):
        """Creates an instance of the class.

        Args:
            datasets (Dataset | Iterable[Dataset]): Dataset(s) to be wrapped by
                this data pipeline.
            preprocess_fn (Callable[[list[DictData]], list[DictData]]):
                Preprocessing function of a single sample. It takes a list of
                samples and returns a list of samples. Defaults to identity
                function.
        """
        if isinstance(datasets, Dataset):
            datasets = [datasets]
        super().__init__(datasets)
        self.preprocess_fn = preprocess_fn

        if any(isinstance(dataset, MultiViewDataset) for dataset in datasets):
            if not all(
                isinstance(dataset, MultiViewDataset) for dataset in datasets
            ):
                raise ValueError(
                    "All datasets must be MultiViewDataset if one of them is."
                )
            self.has_reference = True
        else:
            self.has_reference = False

    def __getitem__(self, idx: int) -> DictDataOrList:
        """Wrap getitem to apply augmentations."""
        samples = super().__getitem__(idx)
        if isinstance(samples, list):
            return self.preprocess_fn(samples)

        return self.preprocess_fn([samples])[0]


class MosaicDataPipe(DataPipe):
    """MosaicDataPipe class.

    This class wraps DataPipe to support mosaic augmentation by sampling three
    additional indices for each image.
    """

    def _sample_mosaic_indices(self, idx: int, data_len: int) -> list[int]:
        """Sample indices for mosaic augmentation."""
        indices = [idx]
        for _ in range(1, 4):
            rand_ind = random.randint(0, data_len - 1)
            while rand_ind in indices:
                rand_ind = random.randint(0, data_len - 1)
            indices.append(rand_ind)
        return indices

    def __getitem__(self, idx: int) -> DictDataOrList:
        """Wrap getitem to apply augmentations."""
        samples = super(DataPipe, self).__getitem__(idx)
        if isinstance(samples, list):
            # TODO: Implement mosaic augmentation for multi-view datasets.
            return self.preprocess_fn(samples)

        mosaic_inds = self._sample_mosaic_indices(idx, len(self))
        prep_samples = self.preprocess_fn(
            [samples]
            + [
                super(DataPipe, self).__getitem__(ind)
                for ind in mosaic_inds[1:]
            ]
        )
        return prep_samples[0]
