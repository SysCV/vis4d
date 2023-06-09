"""Dataloader utility functions."""
from __future__ import annotations

from collections.abc import Callable

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler

from vis4d.common.distributed import get_world_size

from .const import CommonKeys as K
from .data_pipe import DataPipe
from .samplers import VideoInferenceSampler
from .transforms import compose
from .transforms.to_tensor import ToTensor
from .typing import DictData, DictDataOrList


def default_collate(batch: list[DictData]) -> DictData:
    """Default batch collate.

    It will concatenate images and stack seg_masks, extrinsics, intrinsics,
    and depth_maps. Other keys will be put into a list.

    Args:
        batch (list[DictData]): List of data dicts.

    Returns:
        DictData: Collated data dict.
    """
    data: DictData = {}
    # TODO: Add Union key support
    for key in batch[0]:
        try:
            if key in [K.images]:
                data[key] = torch.cat([b[key] for b in batch])
            elif key in [
                K.seg_masks,
                K.extrinsics,
                K.intrinsics,
                K.depth_maps,
                K.optical_flows,
            ]:
                data[key] = torch.stack([b[key] for b in batch], 0)
            else:
                data[key] = [b[key] for b in batch]
        except RuntimeError as e:
            raise RuntimeError(f"Error collating key {key}") from e
    return data


def multi_sensor_collate(batch: list[DictData]) -> DictData:
    """Default multi-sensor batch collate.

    Args:
        batch (list[DictData]): List of data dicts. Each data dict contains
            data from multiple sensors.

    Returns:
        DictData: Collated data dict.
    """
    collated_batch = {}
    sensors = list(batch[0].keys())

    # For each sensor, collate the batch.
    for sensor in sensors:
        # Only collate if all items are a dict, otherwise keep as it is.
        inner_batch = [b[sensor] for b in batch]
        if all(isinstance(item, dict) for item in inner_batch):
            collated_batch[sensor] = default_collate(inner_batch)
        else:
            collated_batch[sensor] = inner_batch

    return collated_batch


def default_pipeline(data: list[DictData]) -> list[DictData]:
    """Default data pipeline."""
    return compose([ToTensor()])(data)


def build_train_dataloader(
    dataset: DataPipe | IterableDataset[DictData],
    samples_per_gpu: int = 1,
    workers_per_gpu: int = 1,
    batchprocess_fn: Callable[
        [list[DictData]], list[DictData]
    ] = default_pipeline,
    collate_fn: Callable[[list[DictData]], DictData] = default_collate,
    pin_memory: bool = True,
    shuffle: bool = True,
) -> DataLoader[DictDataOrList]:
    """Build training dataloader."""

    def _collate_fn_single(data: list[DictData]) -> DictDataOrList:
        """Collates data from single view dataset."""
        return collate_fn(batch=batchprocess_fn(data))  # type: ignore

    def _collate_fn_multi(data: list[list[DictData]]) -> DictDataOrList:
        """Collates data from multi view dataset."""
        views = []
        for view_idx in range(len(data[0])):
            view = collate_fn(
                batch=batchprocess_fn([d[view_idx] for d in data])  # type: ignore # pylint: disable=line-too-long
            )
            views.append(view)
        return views

    sampler = None
    if get_world_size() > 1:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False

    if isinstance(dataset, DataPipe) and dataset.has_reference:
        _collate_fn = _collate_fn_multi
    else:
        _collate_fn = _collate_fn_single

    dataloader = DataLoader(
        dataset,
        batch_size=samples_per_gpu,
        num_workers=workers_per_gpu,
        collate_fn=_collate_fn,
        sampler=sampler,
        persistent_workers=workers_per_gpu > 0,
        pin_memory=pin_memory,
        shuffle=shuffle,
    )
    return dataloader


def build_inference_dataloaders(
    datasets: Dataset[DictDataOrList] | list[Dataset[DictDataOrList]],
    samples_per_gpu: int = 1,
    workers_per_gpu: int = 1,
    video_based_inference: bool = False,
    batchprocess_fn: Callable[
        [list[DictData]], list[DictData]
    ] = default_pipeline,
    collate_fn: Callable[[list[DictData]], DictData] = default_collate,
) -> list[DataLoader[DictDataOrList]]:
    """Build dataloaders for test / predict."""

    def _collate_fn(data: list[DictData]) -> DictDataOrList:
        """Collates data for inference."""
        return collate_fn(batch=batchprocess_fn(data))  # type: ignore

    if isinstance(datasets, Dataset):
        datasets_ = [datasets]
    else:
        datasets_ = datasets

    dataloaders = []
    for dataset in datasets_:
        if isinstance(dataset, DataPipe):
            assert (
                len(dataset.datasets) == 1
            ), "Inference needs a single dataset per DataPipe."
            current_dataset = dataset.datasets[0]
        else:
            current_dataset = dataset

        sampler: DistributedSampler[list[int]] | None
        if get_world_size() > 1:
            if video_based_inference:
                assert hasattr(current_dataset, "video_to_indices"), (
                    "Need video_to_indices attribute for VideoInferenceSampler"
                    " to split dataset by sequences!"
                )
                sampler = VideoInferenceSampler(current_dataset)
            else:
                sampler = DistributedSampler(dataset)
        else:
            sampler = None

        test_dataloader = DataLoader(
            dataset,
            batch_size=samples_per_gpu,
            num_workers=workers_per_gpu,
            sampler=sampler,
            shuffle=False,
            collate_fn=_collate_fn,
            persistent_workers=workers_per_gpu > 0,
        )
        dataloaders.append(test_dataloader)
    return dataloaders
