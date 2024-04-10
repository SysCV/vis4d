"""Iterable datasets."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator

import numpy as np
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from .typing import DictData


class SubdividingIterableDataset(IterableDataset[DictData]):
    """Subdivides a given dataset into smaller chunks.

    This also adds a field called 'index' (DataKeys.index) to the data
    struct in order to relate the data to the source index.

    Example: Given a dataset (ds) that outputs tensors of the shape (10, 3):
    sub_ds = SubdividingIterableDataset(ds, n_samples_per_batch = 5)

    next(iter(sub_ds))['key'].shape
    >> torch.Size([5, 3])

    next(DataLoader(sub_ds, batch_size = 4))['key'].shape
    >> torch.size([4,5,3])

    Assuming the dataset returns two entries with shape (10,3):
    [e['index'].item() for e in sub_ds]
    >> [0,0,1,1]
    """

    def __init__(
        self,
        dataset: Dataset[DictData],
        n_samples_per_batch: int,
        preprocess_fn: Callable[
            [list[DictData]], list[DictData]
        ] = lambda x: x,
    ) -> None:
        """Creates a new Dataset.

        Args:
            dataset (Dataset): The dataset which should be subdivided.
            n_samples_per_batch: How many samples each batch should contain.
                The first dimension of dataset[0].shape must be divisible by
                this number.
            preprocess_fn (Callable[[list[DictData]], list[DictData]):
                Preprocessing function. Defaults to identity.
        """
        super().__init__()
        self.dataset = dataset
        self.n_samples_per_batch = n_samples_per_batch
        self.preprocess_fn = preprocess_fn

    def __getitem__(self, index: int) -> DictData:
        """Indexing is not supported for IterableDatasets."""
        raise NotImplementedError("IterableDataset does not support indeing")

    def __iter__(self) -> Iterator[DictData]:
        """Iterates over the dataset, supporting distributed sampling."""
        worker_info = get_worker_info()
        if worker_info is None:
            # not distributed
            num_workers = 1
            worker_id = 0
        else:  # pragma: no cover
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

        assert hasattr(
            self.dataset, "__len__"
        ), "Dataset must have __len__ in order to be subdivided."
        n_samples = len(self.dataset)
        for i in range(math.ceil(n_samples / num_workers)):
            data_idx = i * num_workers + worker_id
            if data_idx >= n_samples:
                continue
            data_sample = self.dataset[data_idx]

            n_elements = list((data_sample.values()))[0].shape[0]
            for idx in range(int(n_elements / self.n_samples_per_batch)):
                # This is kind of ugly
                # this field defines from which source the data was loaded
                # (first entry, second entry, ...)
                # this is required if we e.g. want to subdivide a room that is
                # too big into equal sized chunks and stick them back together
                # for visualizaton
                out_data: DictData = {"source_index": np.ndarray([data_idx])}
                for key in data_sample:
                    start_idx = idx * self.n_samples_per_batch
                    end_idx = (idx + 1) * self.n_samples_per_batch
                    if (len(data_sample[key])) < self.n_samples_per_batch:
                        out_data[key] = data_sample[key]
                    else:
                        out_data[key] = data_sample[key][
                            start_idx:end_idx, ...
                        ]
                yield self.preprocess_fn([out_data])[0]
