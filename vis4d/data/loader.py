"""Dataloader utility functions."""

from __future__ import annotations

import random
import warnings
from collections.abc import Callable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from vis4d.common.distributed import get_rank, get_world_size

from .const import CommonKeys as K
from .data_pipe import DataPipe
from .datasets import VideoDataset
from .samplers import VideoInferenceSampler
from .transforms import compose
from .transforms.to_tensor import ToTensor
from .typing import DictData, DictDataOrList

DEFAULT_COLLATE_KEYS = (
    K.seg_masks,
    K.extrinsics,
    K.intrinsics,
    K.depth_maps,
    K.optical_flows,
    K.categories,
)


def default_collate(
    batch: list[DictData],
    collate_keys: Sequence[str] = DEFAULT_COLLATE_KEYS,
    sensors: Sequence[str] | None = None,
) -> DictData:
    """Default batch collate.

    It will concatenate images and stack seg_masks, extrinsics, intrinsics,
    and depth_maps. Other keys will be put into a list.

    Args:
        batch (list[DictData]): List of data dicts.
        collate_keys (Sequence[str]): Keys to be collated. Default is
            DEFAULT_COLLATE_KEYS.
        sensors (Sequence[str] | None): List of sensors to collate. If is not
            None will raise an error. Default is None.

    Returns:
        DictData: Collated data dict.
    """
    assert sensors is None, "If specified sensors, use multi_sensor_collate."

    data: DictData = {}
    for key in batch[0]:
        try:
            if key == "transforms":  # skip transform parameters
                continue
            if key in [K.images]:
                data[key] = torch.cat([b[key] for b in batch])
            elif key in collate_keys:
                data[key] = torch.stack([b[key] for b in batch], 0)
            else:
                data[key] = [b[key] for b in batch]
        except RuntimeError as e:
            raise RuntimeError(f"Error collating key {key}") from e
    return data


def multi_sensor_collate(
    batch: list[DictData],
    collate_keys: Sequence[str] = DEFAULT_COLLATE_KEYS,
    sensors: Sequence[str] | None = None,
) -> DictData:
    """Default multi-sensor batch collate.

    Args:
        batch (list[DictData]): List of data dicts. Each data dict contains
            data from multiple sensors.
        collate_keys (Sequence[str]): Keys to be collated. Default is
            DEFAULT_COLLATE_KEYS.
        sensors (Sequence[str] | None): List of sensors to collate. If None,
            will raise an error. Default is None.

    Returns:
        DictData: Collated data dict.
    """
    assert (
        sensors is not None
    ), "If not specified sensors, use default_collate."

    collated_batch: DictData = {}

    # For each sensor, collate the batch. Other keys will be put into a list.
    for key in batch[0]:
        inner_batch = [b[key] for b in batch]
        if key in sensors:
            collated_batch[key] = default_collate(inner_batch, collate_keys)
        else:
            collated_batch[key] = inner_batch
    return collated_batch


def default_pipeline(data: list[DictData]) -> list[DictData]:
    """Default data pipeline."""
    return compose([ToTensor()])(data)


def build_train_dataloader(
    dataset: DataPipe,
    samples_per_gpu: int = 1,
    workers_per_gpu: int = 1,
    batchprocess_fn: Callable[
        [list[DictData]], list[DictData]
    ] = default_pipeline,
    collate_fn: Callable[
        [list[DictData], Sequence[str]], DictData
    ] = default_collate,
    collate_keys: Sequence[str] = DEFAULT_COLLATE_KEYS,
    sensors: Sequence[str] | None = None,
    pin_memory: bool = True,
    shuffle: bool = True,
    seed: int | None = None,
    disable_subprocess_warning: bool = False,
) -> DataLoader[DictDataOrList]:
    """Build training dataloader."""
    assert isinstance(dataset, DataPipe), "dataset must be a DataPipe"

    def _collate_fn_single(data: list[DictData]) -> DictData:
        """Collates data from single view dataset."""
        return collate_fn(  # type: ignore
            batch=batchprocess_fn(data),
            collate_keys=collate_keys,
            sensors=sensors,
        )

    def _collate_fn_multi(data: list[list[DictData]]) -> list[DictData]:
        """Collates data from multi view dataset."""
        views = []
        for view_idx in range(len(data[0])):
            view = collate_fn(  # type: ignore
                batch=batchprocess_fn([d[view_idx] for d in data]),
                collate_keys=collate_keys,
                sensors=sensors,
            )
            views.append(view)
        return views

    def _worker_init_fn(worker_id: int) -> None:
        """Will be called on each worker after seeding and before data loading.

        Args:
            worker_id (int): Worker id in [0, num_workers - 1].
        """
        if seed is not None:
            # The seed of each worker equals to
            # num_workers * rank + worker_id + user_seed
            worker_seed = workers_per_gpu * get_rank() + worker_id + seed
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)
            if disable_subprocess_warning and worker_id != 0:
                warnings.simplefilter("ignore")

    sampler = None
    if get_world_size() > 1:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False

    dataloader = DataLoader(
        dataset,
        batch_size=samples_per_gpu,
        num_workers=workers_per_gpu,
        collate_fn=(
            _collate_fn_multi if dataset.has_reference else _collate_fn_single
        ),
        sampler=sampler,
        worker_init_fn=_worker_init_fn,
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
    collate_fn: Callable[
        [list[DictData], Sequence[str]], DictData
    ] = default_collate,
    collate_keys: Sequence[str] = DEFAULT_COLLATE_KEYS,
    sensors: Sequence[str] | None = None,
) -> list[DataLoader[DictDataOrList]]:
    """Build dataloaders for test / predict."""

    def _collate_fn(data: list[DictData]) -> DictData:
        """Collates data for inference."""
        return collate_fn(  # type: ignore
            batch=batchprocess_fn(data),
            collate_keys=collate_keys,
            sensors=sensors,
        )

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
                assert isinstance(
                    current_dataset, VideoDataset
                ), "Video based inference needs a VideoDataset."
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
