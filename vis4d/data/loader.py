"""Dataloader utility functions."""
from __future__ import annotations

import bisect
import math
from collections.abc import Callable, Iterable, Iterator
from typing import Union

import numpy as np
import torch
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    IterableDataset,
    get_worker_info,
)
from torch.utils.data.distributed import DistributedSampler

from vis4d.common import ArgsType

from ..common.distributed import get_world_size
from .const import CommonKeys as K
from .datasets import VideoMixin
from .reference import ReferenceViewSampler
from .samplers import VideoInferenceSampler
from .transforms import compose_batch
from .transforms.to_tensor import ToTensor
from .typing import DictData

DictDataOrList = Union[DictData, list[DictData]]
_DATASET = Dataset[DictDataOrList]
_DATASET_ONLY_DICT = Dataset[DictData]  # pylint: disable=invalid-name
_CONCAT_DATASET = ConcatDataset[DictDataOrList]  # pylint: disable=invalid-name
_ITERABLE_DATASET = IterableDataset[DictData]  # pylint: disable=invalid-name
_DATALOADER = DataLoader[DictDataOrList]  # pylint: disable=invalid-name


def default_collate(batch: list[DictData]) -> DictData:
    """Default batch collate."""
    data: DictData = {}
    # TODO: It seems dangerous if batches originally contain different keys.
    # e.g. if batch[0] has annotations but batch[1] doesn't.
    for key in batch[0]:
        try:
            if key in [K.images]:
                data[key] = torch.cat([b[key] for b in batch])
            elif key in [K.segmentation_masks, K.extrinsics, K.intrinsics]:
                data[key] = torch.stack([b[key] for b in batch], 0)
            else:
                data[key] = [b[key] for b in batch]
        except RuntimeError as e:
            raise RuntimeError(f"Error collating key {key}") from e
    return data


def multi_sensor_collate(batch: list[DictData]) -> DictData:
    """Default multi-sensor batch collate."""
    data = {}
    sensors = list(batch[0].keys())
    for sensor in sensors:
        data[sensor] = default_collate([d[sensor] for d in batch])
    return data


class DataPipe(_CONCAT_DATASET):
    """DataPipe class.

    This class wraps one or multiple instances of a PyTorch Dataset so that the
    preprocessing steps can be shared across those datasets. Composes dataset
    and the preprocessing pipeline.
    """

    def __init__(
        self,
        datasets: _DATASET | Iterable[_DATASET],
        preprocess_fn: Callable[[DictData], DictData] = lambda x: x,
        reference_view_sampler: None | ReferenceViewSampler = None,
    ):
        """Creates an instance of the class.

        Args:
            datasets (Dataset | Iterable[Dataset]): Dataset(s) to be
                wrapped by this data pipeline.
            preprocess_fn (Callable[[DataDict], DataDict]): Preprocessing
                function of a single sample.
            reference_view_sampler (None | ReferenceViewSampler, optional): For
                video datasets, the reference sampler decides how reference
                views will be sampled. Used for training, e.g., tracking models
                on multiple frames of a video. Defaults to None.
        """
        if isinstance(datasets, Dataset):
            datasets = [datasets]
        super().__init__(datasets)
        self.preprocess_fn = preprocess_fn
        self.reference_view_sampler = reference_view_sampler

    def get_dataset_sample_index(self, idx: int) -> tuple[int, int]:
        """Get dataset and sample index from global index."""
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return dataset_idx, sample_idx

    def _getitem(self, idx: int) -> DictDataOrList:
        """Modular re-implementation of getitem."""
        dataset_idx, sample_idx = self.get_dataset_sample_index(idx)
        return self.datasets[dataset_idx][sample_idx]

    def __getitem__(self, idx: int) -> DictDataOrList:
        """Wrap getitem to apply augmentations."""
        if self.reference_view_sampler is not None:
            dataset_idx, _ = self.get_dataset_sample_index(idx)
            dataset = self.datasets[dataset_idx]
            assert isinstance(dataset, VideoMixin), (
                "Reference view sampling is only supported for datasets that "
                "implement the VideoMixin. Incompatible dataset: "
                f"{self.datasets[dataset_idx]}"
            )
            video_indices = dataset.get_video_indices(idx)
            indices = self.reference_view_sampler(idx, video_indices)
            samples = [self._getitem(i) for i in indices]
            # TODO re-use transform parameters across reference views.
            data = [self.preprocess_fn(sample) for sample in samples]
        else:
            sample = self._getitem(idx)
            data = self.preprocess_fn(sample)
        return data


# TODO: refactor data pipe with mixin
class VideoDataPipe(DataPipe, VideoMixin):
    """DataPipe class for video datasets."""

    def __init__(self, *args: ArgsType, **kwargs: ArgsType) -> None:
        """Create data pipe for video datasets."""
        super().__init__(*args, **kwargs)
        assert len(self.datasets) == 1, "Only support one dataset for now."

    @property
    def video_to_indices(self) -> dict[str, list[int]]:
        """Get video to indices mapping."""
        return self.datasets[0].video_to_indices


class SubdividingIterableDataset(_ITERABLE_DATASET):
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
        dataset: _DATASET_ONLY_DICT,
        n_samples_per_batch: int,
        preprocess_fn: Callable[[DictData], DictData] | None = None,
    ) -> None:
        """Creates a new Dataset.

        Args:
            dataset (Dataset): The dataset which should be subdivided.
            n_samples_per_batch: How many samples each batch should contain.
                The first dimension of dataset[0].shape must be divisible by
                this number.
            preprocess_fn (Callable[[DataDict], DataDict]): Preprocessing
                function of a single sample. Can be None.
        """
        super().__init__()
        self.dataset = dataset
        self.n_samples_per_batch = n_samples_per_batch
        self.preprocess_fn = preprocess_fn
        self.reference_view_sampler = None

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
        else:
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
                out_data = {"source_index": np.ndarray([data_idx])}
                for key in data_sample:
                    start_idx = idx * self.n_samples_per_batch
                    end_idx = (idx + 1) * self.n_samples_per_batch
                    if (len(data_sample[key])) < self.n_samples_per_batch:
                        out_data[key] = data_sample[key]
                    else:
                        out_data[key] = data_sample[key][
                            start_idx:end_idx, ...
                        ]
                yield self.preprocess_fn(
                    out_data
                ) if self.preprocess_fn else out_data


def default_pipeline(data: list[DictData]) -> DictData:
    """Default data pipeline."""
    return compose_batch([ToTensor()])(data)


def build_train_dataloader(
    dataset: DataPipe,
    samples_per_gpu: int = 1,
    workers_per_gpu: int = 1,
    batchprocess_fn: Callable[
        [list[DictData]], list[DictData]
    ] = default_pipeline,
    collate_fn: Callable[[list[DictData]], DictData] = default_collate,
    pin_memory: bool = True,
    shuffle: bool = True,
) -> _DATALOADER:
    """Build training dataloader."""

    def _collate_fn_single(data: list[DictData]) -> DictDataOrList:
        """Collates data from single view dataset."""
        return collate_fn(batchprocess_fn(data))

    def _collate_fn_multi(data: list[list[DictData]]) -> DictDataOrList:
        """Collates data from multi view dataset."""
        views = []
        for view_idx in range(len(data[0])):
            view = collate_fn(batchprocess_fn([d[view_idx] for d in data]))
            views.append(view)
        return views

    sampler = None
    if get_world_size() > 1:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False

    dataloader = DataLoader(
        dataset,
        batch_size=samples_per_gpu,
        num_workers=workers_per_gpu,
        collate_fn=_collate_fn_single
        if dataset.reference_view_sampler is None
        else _collate_fn_multi,
        sampler=sampler,
        persistent_workers=workers_per_gpu > 0,
        pin_memory=pin_memory,
        shuffle=shuffle,
    )
    return dataloader


def build_inference_dataloaders(
    datasets: _DATASET | list[_DATASET],
    samples_per_gpu: int = 1,
    workers_per_gpu: int = 1,
    video_based_inference: bool = True,
    batchprocess_fn: Callable[
        [list[DictData]], list[DictData]
    ] = default_pipeline,
    collate_fn: Callable[[list[DictData]], DictData] = default_collate,
) -> list[_DATALOADER]:
    """Build dataloaders for test / predict."""

    def _collate_fn(data: list[DictData]) -> DictDataOrList:
        """Collates data for inference."""
        return collate_fn(batchprocess_fn(data))

    if isinstance(datasets, Dataset):
        datasets_ = [datasets]
    else:
        datasets_ = datasets
    dataloaders = []
    for dataset in datasets_:
        dset_sampler: DistributedSampler[list[int]] | None
        if get_world_size() > 1:
            if isinstance(dataset, VideoMixin) and video_based_inference:
                dset_sampler = VideoInferenceSampler(dataset)  # type: ignore
            else:
                dset_sampler = DistributedSampler(dataset)
        else:
            dset_sampler = None

        test_dataloader = DataLoader(
            dataset,
            batch_size=samples_per_gpu,
            num_workers=workers_per_gpu,
            sampler=dset_sampler,
            shuffle=False,
            collate_fn=_collate_fn,
            persistent_workers=workers_per_gpu > 0,
        )
        dataloaders.append(test_dataloader)
    return dataloaders
