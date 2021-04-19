"""Build data loading pipeline for tracking."""
import logging
from typing import Any, Dict, List, Optional, Union

import torch
from detectron2.config import CfgNode
from detectron2.data import (
    build_batch_data_loader,
    get_detection_dataset_dicts,
)
from detectron2.data.common import DatasetFromList
from detectron2.data.samplers import (
    RepeatFactorTrainingSampler,
    TrainingSampler,
)
from pydantic import BaseModel

from openmt.common.io import DataBackendConfig

from .dataset_mapper import (
    MapTrackingDataset,
    ReferenceSamplingConfig,
    TrackingDatasetMapper,
)
from .samplers import TrackingInferenceSampler


class DataloaderConfig(BaseModel):
    """Config for dataloader."""

    data_backend: DataBackendConfig = DataBackendConfig()
    num_workers: int
    sampling_cfg: ReferenceSamplingConfig


class DataOptions(BaseModel):
    """Options for building the dataloader."""

    dataset: List[Dict[str, Any]]  # type: ignore
    sampler: Optional[Union[TrainingSampler, RepeatFactorTrainingSampler]]
    mapper: TrackingDatasetMapper
    total_batch_size: int
    num_workers: int

    class Config:  # needed due to sampler
        """Pydantic configuration for this particular class."""

        arbitrary_types_allowed = True


def _train_loader_from_config(
    loader_cfg: DataloaderConfig, cfg: CfgNode
) -> DataOptions:
    """Construct training data loader from config."""
    dataset = get_detection_dataset_dicts(
        cfg.DATASETS.TRAIN,
        filter_empty=False,
        min_keypoints=0,
        proposal_files=None,
    )

    mapper = TrackingDatasetMapper(loader_cfg.data_backend, cfg)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler %s", sampler_name)
    if sampler_name == "TrainingSampler":
        sampler = TrainingSampler(len(dataset))
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))

    return DataOptions(
        dataset=dataset,
        sampler=sampler,
        mapper=mapper,
        total_batch_size=cfg.SOLVER.IMS_PER_BATCH,
        num_workers=loader_cfg.num_workers,
    )


def build_tracking_train_loader(
    loader_cfg: DataloaderConfig, det2cfg: CfgNode
) -> torch.utils.data.DataLoader:
    """Build train dataloader for tracking with some default features."""
    data_options = _train_loader_from_config(loader_cfg, det2cfg)
    dataset = DatasetFromList(data_options.dataset, copy=False)
    dataset = MapTrackingDataset(
        loader_cfg.sampling_cfg, True, dataset, data_options.mapper
    )
    assert isinstance(data_options.sampler, torch.utils.data.sampler.Sampler)
    # aspect_ratio_grouping: tracking datasets usually do not contain
    # sequences with vastly different aspect ratios --> set to False
    return build_batch_data_loader(
        dataset,
        data_options.sampler,
        data_options.total_batch_size,
        aspect_ratio_grouping=False,
        num_workers=data_options.num_workers,
    )


def _test_loader_from_config(
    loader_cfg: DataloaderConfig, cfg: CfgNode, dataset_name: str
) -> DataOptions:
    """Construct testing data loader from config."""
    dataset = get_detection_dataset_dicts(
        dataset_name,
        filter_empty=False,
        min_keypoints=0,
        proposal_files=None,
    )
    mapper = TrackingDatasetMapper(loader_cfg.data_backend, cfg)
    return DataOptions(
        dataset=dataset,
        mapper=mapper,
        total_batch_size=1,
        num_workers=loader_cfg.num_workers,
    )


def build_tracking_test_loader(
    loader_cfg: DataloaderConfig, det2cfg: CfgNode, dataset_name: str
) -> torch.utils.data.DataLoader:
    """Build test dataloader for tracking with some default features."""
    data_options = _test_loader_from_config(loader_cfg, det2cfg, dataset_name)
    dataset = DatasetFromList(data_options.dataset, copy=False)
    dataset = MapTrackingDataset(
        loader_cfg.sampling_cfg, False, dataset, data_options.mapper
    )
    sampler = TrackingInferenceSampler(dataset)
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, 1, drop_last=False
    )
    return torch.utils.data.DataLoader(
        dataset,
        num_workers=data_options.num_workers,
        batch_sampler=batch_sampler,
        collate_fn=lambda x: x,
    )
