"""Build data loading pipeline for tracking."""
import logging
from typing import Dict, List, Union

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

from .dataset_mapper import (
    MapTrackingDataset,
    ReferenceSamplingConfig,
    TrackingDatasetMapper,
)
from .io import DataBackendConfig


class DataloaderConfig(BaseModel):
    """Config for dataloader."""

    data_backend: DataBackendConfig = DataBackendConfig()
    num_workers: int
    sampling_cfg: ReferenceSamplingConfig


class DataOptions(BaseModel):
    """Options for building the dataloader."""

    dataset: List[Dict]
    sampler: Union[TrainingSampler, RepeatFactorTrainingSampler]
    mapper: TrackingDatasetMapper
    total_batch_size: int
    num_workers: int


def _train_loader_from_config(
    loader_cfg: DataloaderConfig, cfg: CfgNode
) -> DataOptions:
    """Construct training data loader from config."""
    dataset = get_detection_dataset_dicts(
        cfg.DATASETS.TRAIN,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.MODEL.KEYPOINT_ON
        else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )

    mapper = TrackingDatasetMapper(loader_cfg.data_backend, cfg)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler %s", sampler_name)
    if sampler_name == "TrainingSampler":
        sampler = TrainingSampler(len(dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        repeat_factors = (
            RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
                dataset, cfg.DATALOADER.REPEAT_THRESHOLD
            )
        )
        sampler = RepeatFactorTrainingSampler(repeat_factors)
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
):
    """Build a dataloader for object tracking with some default features.
    This interface is experimental.
    """
    data_options = _train_loader_from_config(loader_cfg, det2cfg)
    dataset = DatasetFromList(data_options.dataset, copy=False)
    dataset = MapTrackingDataset(
        loader_cfg.sampling_cfg, dataset, data_options.mapper
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
