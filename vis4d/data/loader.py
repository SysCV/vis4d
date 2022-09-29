"""Dataloader utility functions."""
from typing import Callable, Iterable, List, Optional, Union

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from vis4d.data.samplers import BaseSampler, VideoInferenceSampler

from ..common_to_revise.utils import get_world_size
from .datasets import BaseVideoDataset
from .datasets.base import DataKeys, DictData


def default_collate(batch: List[DictData]) -> DictData:
    """Default batch collate."""
    data = {}
    for key in batch[0]:
        if key == DataKeys.images:
            data[key] = torch.cat([b[key] for b in batch])
        elif key == DataKeys.metadata:
            data[key] = {k: [b[key][k] for b in batch] for k in batch[0][key]}
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
        data, params = self.preprocess_fn(sample)
        return data


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
            and isinstance(dataset, BaseVideoDataset)
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
