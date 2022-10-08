"""Dataloader utility functions."""
import math
from typing import Callable, Iterable, Iterator, List, Optional, Union

import torch
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    IterableDataset,
    get_worker_info,
)

from vis4d.data.samplers import BaseSampler, VideoInferenceSampler

from ..common.utils import get_world_size
from .datasets import VideoDataset
from .datasets.base import COMMON_KEYS, DictData

"""Keys that contain pointcloud based data and can be stacked using torch.stack."""
POINT_KEYS = [
    COMMON_KEYS.colors3d,
    COMMON_KEYS.points3d,
    COMMON_KEYS.points3dCenter,
    COMMON_KEYS.semantics3d,
    COMMON_KEYS.instances3d,
    COMMON_KEYS.index,
]


def default_collate(batch: List[DictData]) -> DictData:
    """Default batch collate."""
    data = {}
    for key in batch[0]:
        if key == COMMON_KEYS.images:
            data[key] = torch.cat([b[key] for b in batch])
        elif key == COMMON_KEYS.metadata:
            data[key] = {k: [b[key][k] for b in batch] for k in batch[0][key]}
        elif key in POINT_KEYS:
            data[key] = torch.stack([b[key] for b in batch], 0)
        else:
            data[key] = [b[key] for b in batch]
    return data


class DataPipe(ConcatDataset):
    """DataPipe class.

    This class wraps one or multiple instances of a PyTorch Dataset so that the
    preprocessing steps can be shared across those datasets.
    """

    def __init__(
        self,
        datasets: Union[Dataset, Iterable[Dataset]],
        preprocess_fn: Callable[[DictData], DictData],
    ):
        """Init.

        Args:
            datasets (Union[Dataset, Iterable[Dataset]]): Dataset(s) to be wrapped by this data pipeline.
            preprocess_fn (Callable[[DataDict], DataDict]): Preprocessing function of a single sample.
        """
        if isinstance(datasets, Dataset):
            datasets = [datasets]
        super().__init__(datasets)
        self.preprocess_fn = preprocess_fn

    def __getitem__(self, idx: int) -> DictData:
        """Wrap getitem to apply augmentations."""
        getitem = super().__getitem__
        sample = getitem(idx)
        data = self.preprocess_fn(sample)
        return data


class SubdividingIterableDataset(IterableDataset):
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
        self, dataset: Dataset[DictData], n_samples_per_batch: int
    ) -> None:
        """Creates a new Dataset
        Args:
            dataset (Dataset): The dataset which should be subdivided
            n_samples_per_batch: How many samples each batch should contain.
                                 The first dimension of dataset[0].shape must
                                 be divisible by this number
        """
        super().__init__()

        self.dataset = dataset
        self.n_samples_per_batch = n_samples_per_batch

    def __iter__(self) -> Iterator[DictData]:
        """Iterates over the dataset, supporting distributed sampling."""
        worker_info = get_worker_info()
        if worker_info is None:
            # not distributed
            num_workers = 1
            worker_id = 0
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

        for i in range(math.ceil(len(self.dataset) / num_workers)):
            data_idx = i * num_workers + worker_id
            if data_idx < len(self.dataset):
                data_sample = self.dataset[data_idx]
                n_elements = next(iter(data_sample.values())).size(0)
                for idx in range(int(n_elements / self.n_samples_per_batch)):
                    out_data = {COMMON_KEYS.index: torch.tensor([data_idx])}
                    for key in data_sample:
                        start_idx = idx * self.n_samples_per_batch
                        end_idx = (idx + 1) * self.n_samples_per_batch
                        out_data[key] = data_sample[key][
                            start_idx:end_idx, ...
                        ]
                    yield out_data


def build_train_dataloader(
    dataset: Dataset,
    samples_per_gpu: int = 1,
    workers_per_gpu: int = 1,
    batchprocess_fn: Callable[[List[DictData]], List[DictData]] = lambda x: x,
    collate_fn: Callable[[List[DictData]], DictData] = default_collate,
    sampler: Optional[BaseSampler] = None,
    pin_memory: bool = True,
) -> DataLoader:
    """Build training dataloader."""
    if sampler is not None:
        batch_size, shuffle = 1, False
    else:
        batch_size, shuffle, train_sampler = (
            samples_per_gpu,
            True,
            None,
        )
    dataloader = DataLoader(
        dataset,
        batch_sampler=train_sampler,
        batch_size=batch_size,
        num_workers=workers_per_gpu,
        collate_fn=lambda x: collate_fn(batchprocess_fn(x)),
        persistent_workers=workers_per_gpu > 0,
        pin_memory=pin_memory,
        shuffle=shuffle,
    )
    return dataloader


def build_inference_dataloaders(
    datasets: Union[Dataset, List[Dataset]],
    samples_per_gpu: int = 1,
    workers_per_gpu: int = 1,
    video_based_inference: bool = True,
    batchprocess_fn: Callable[[List[DictData]], List[DictData]] = lambda x: x,
    collate_fn: Callable[[List[DictData]], DictData] = default_collate,
    sampler: Optional[BaseSampler] = None,
) -> List[DataLoader]:
    """Build dataloaders for test / predict."""
    if isinstance(datasets, Dataset):
        datasets = [datasets]
    dataloaders = []
    for dataset in datasets:
        if (
            get_world_size() > 1
            and isinstance(dataset, VideoDataset)
            and video_based_inference
        ):
            sampler = VideoInferenceSampler(dataset)

        test_dataloader = DataLoader(
            dataset,
            batch_size=samples_per_gpu,
            num_workers=workers_per_gpu,
            sampler=sampler,
            collate_fn=lambda x: collate_fn(batchprocess_fn(x)),
            persistent_workers=workers_per_gpu > 0,
        )
        dataloaders.append(test_dataloader)
    return dataloaders
