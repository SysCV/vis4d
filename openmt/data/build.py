"""Build data loading pipeline for tracking."""
import logging

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
from detectron2.utils.logger import _log_api_usage
from pydantic import BaseModel

from .dataset_mapper import MapTrackingDataset, TrackingDatasetMapper
from .io import DataBackendConfig


class DataloaderConfig(BaseModel):
    """Config for dataloader."""
    data_backend: DataBackendConfig = DataBackendConfig()
    num_workers: int


def _train_loader_from_config(loader_cfg: DataloaderConfig, cfg: CfgNode):
    dataset = get_detection_dataset_dicts(
        cfg.DATASETS.TRAIN,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.MODEL.KEYPOINT_ON
        else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
    )
    _log_api_usage("dataset." + cfg.DATASETS.TRAIN[0])

    mapper = TrackingDatasetMapper(loader_cfg.data_backend, cfg)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))
    if sampler_name == "TrainingSampler":
        sampler = TrainingSampler(len(dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
            dataset, cfg.DATALOADER.REPEAT_THRESHOLD
        )
        sampler = RepeatFactorTrainingSampler(repeat_factors)
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))

    return dataset, sampler, mapper, cfg.SOLVER.IMS_PER_BATCH, \
           cfg.DATALOADER.ASPECT_RATIO_GROUPING, loader_cfg.num_workers


def build_tracking_train_loader(loader_cfg: DataloaderConfig, det2cfg: CfgNode):
    """Build a dataloader for object tracking with some default features.
    This interface is experimental.
    """
    dataset, sampler, mapper, total_batch_size, aspect_ratio_grouping, num_workers = _train_loader_from_config(loader_cfg, det2cfg)
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapTrackingDataset(dataset, mapper)
    if sampler is None:
        sampler = TrainingSampler(len(dataset))
    assert isinstance(sampler, torch.utils.data.sampler.Sampler)
    return build_batch_data_loader(
        dataset,
        sampler,
        total_batch_size,
        aspect_ratio_grouping=aspect_ratio_grouping,
        num_workers=num_workers,
    )
