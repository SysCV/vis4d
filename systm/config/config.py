"""Config definitions."""

import os
from datetime import datetime
from typing import List, Optional
from detectron2 import model_zoo
from detectron2.config import get_cfg, CfgNode

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


class Dataset(BaseModel):
    """Config for training/evaluation datasets."""

    name: str


class Config(BaseModel):
    """Overall config object."""

    detection: Detection
    solver: Solver
    train: List[Dataset]
    test: List[Dataset]
    output_dir: Optional[str]


def _prepare_config(config):
    cfg = get_cfg()

    # load model config (either detectron2 or systm)
    if config.detection.base_cfg.startswith('detectron2://'):
        cfg.merge_from_file(model_zoo.get_config_file(config.detection.base_cfg.split('//')[1]))
    elif os.path.exists(config.detection.base_cfg):
        cfg.merge_from_file(config.detection.base_cfg)
    else:
        raise ValueError(f'base config path {config.detection.base_cfg} not found')

    # load checkpoint file
    if config.detection.weights.startswith('detectron2://'):
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config.detection.weights.split('//')[1])
    elif os.path.exists(config.detection.base_cfg):
        cfg.MODEL.WEIGHTS = config.detection.base_cfg
    else:
        raise ValueError(f'model weights path {config.detection.base_cfg} not found')

    # convert model attributes
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = config.detection.num_classes
    cfg.MODEL.RETINANET.NUM_CLASSES = config.detection.num_classes

    # check if output dir variable is filled
    if config.output_dir is not None:
        cfg.OUTPUT_DIR = config.output_dir
    else:
        timestamp = str(datetime.now()).split('.')[0]
        cfg.OUTPUT_DIR = os.path.join('./work_dirs/', config.detection.model_name, timestamp)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg


def read_config(filepath: str) -> CfgNode:
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
    return _prepare_config(config)
