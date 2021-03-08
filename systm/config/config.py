"""Config definitions."""

import os
from datetime import datetime
from enum import Enum
from typing import List, Optional
from detectron2 import model_zoo
from detectron2.config import get_cfg, CfgNode
from detectron2.data.datasets import register_coco_instances


import toml
import yaml
from pydantic import BaseModel


class Solver(BaseModel):
    """Config for solver."""

    images_per_batch: int
    lr_policy: str
    base_lr: float
    steps: List[int]
    max_iters: int


class Detection(BaseModel):
    """Config for detection model training."""

    model_name: str
    base_cfg: str
    weights: Optional[str]
    num_classes: int


class DatasetType(str, Enum):
    """Enum for dataset type.

    coco: COCO style dataset to support detectron2 training.
    custom: Custom dataset type for user-defined datasets.
    """

    COCO = 'coco'
    CUSTOM = 'custom'


class Dataset(BaseModel):
    """Config for training/evaluation datasets."""

    name: str
    type: DatasetType
    data_root: str
    annotation_file: Optional[str]


class Config(BaseModel):
    """Overall config object."""

    detection: Detection
    solver: Solver
    train: List[Dataset]
    test: List[Dataset]
    output_dir: Optional[str]


def _register(datasets: List[Dataset]) -> List[str]:
    """Register dataset in detectron2."""
    names = []
    for dataset in datasets:
        if not dataset.type == DatasetType.COCO:
            raise NotImplementedError("Currently only COCO style dataset "
                                      "structure is supported.")
        register_coco_instances(dataset.name, {}, dataset.annotation_file,
                                dataset.data_root)
        names.append(dataset.name)
    return names


def to_detectron2(config: Config) -> CfgNode:
    """Convert a Config object to a detectron2 readable configuration."""
    cfg = get_cfg()

    # load model config (either detectron2 or systm)
    if config.detection.base_cfg.startswith('detectron2://'):
        base_cfg = model_zoo.get_config_file(
            config.detection.base_cfg.split('//')[1])
        cfg.merge_from_file(base_cfg)
    elif os.path.exists(config.detection.base_cfg):
        cfg.merge_from_file(config.detection.base_cfg)
    else:
        raise ValueError(f'base config path {config.detection.base_cfg} '
                         f'not found')

    # load checkpoint file
    if config.detection.weights is not None:
        if config.detection.weights.startswith('detectron2://'):
            ckpt = config.detection.weights.split('//')[1]
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(ckpt)
        elif os.path.exists(config.detection.base_cfg):
            cfg.MODEL.WEIGHTS = config.detection.base_cfg
        else:
            raise ValueError(f'model weights path {config.detection.base_cfg} '
                             f'not found')

    # convert model attributes
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = config.detection.num_classes
    cfg.MODEL.RETINANET.NUM_CLASSES = config.detection.num_classes
    cfg.OUTPUT_DIR = config.output_dir

    # register datasets
    cfg.DATASETS.TRAIN = _register(config.train)
    cfg.DATASETS.TEST = _register(config.test)
    return cfg


def read_config(filepath: str) -> Config:
    """Read config file and parse it into Config object.

    The config file can be in yaml or toml.
    toml is recommended for readability.
    """
    ext = os.path.splitext(filepath)[1]
    if ext == ".yaml":
        config_dict = yaml.load(
            open(filepath, "r").read(),
            Loader=yaml.Loader,
        )
    elif ext == ".toml":
        config_dict = toml.load(filepath)
    else:
        raise NotImplementedError(f"Config type {ext} not supported")
    config = Config(**config_dict)

    # check if output dir variable is filled, create output dir if necessary
    if config.output_dir is None:
        timestamp = str(datetime.now()).split('.')[0].replace(' ', '_')
        config.output_dir = os.path.join('./work_dirs/',
                                         config.detection.model_name,
                                         timestamp)
    os.makedirs(config.output_dir, exist_ok=True)
    return config
