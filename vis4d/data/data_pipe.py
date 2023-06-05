"""DataPipe wraps datasets to share the prepossessing pipeline."""
from __future__ import annotations

from collections.abc import Callable, Iterable

from torch.utils.data import ConcatDataset, Dataset

from .reference import ReferenceDataset
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
            datasets (Dataset | Iterable[Dataset]): Dataset(s) to be
                wrapped by this data pipeline.
            preprocess_fn (Callable[[DataDict], DataDict]): Preprocessing
                function of a single sample.
        """
        if isinstance(datasets, Dataset):
            datasets = [datasets]
        super().__init__(datasets)
        self.preprocess_fn = preprocess_fn

        if any(
            [isinstance(dataset, ReferenceDataset) for dataset in datasets]
        ):
            if not all(
                [isinstance(dataset, ReferenceDataset) for dataset in datasets]
            ):
                raise ValueError(
                    "All datasets must be ReferenceDataset if one of them is."
                )
            self.has_reference = True
        else:
            self.has_reference = False

    def __getitem__(self, idx: int) -> DictDataOrList:
        """Wrap getitem to apply augmentations."""
        samples = super().__getitem__(idx)
        if isinstance(samples, list):
            data = self.preprocess_fn(samples)
        else:
            data = self.preprocess_fn([samples])[0]

        # TODO: Might need to think about after post-processing transforms,
        # do we need retry mechanism if the annotation is empty?

        return data
