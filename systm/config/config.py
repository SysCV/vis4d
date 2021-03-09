"""Config definitions."""

import os
from datetime import datetime
from enum import Enum
from typing import List, Optional
from detectron2 import model_zoo
from detectron2.config import get_cfg, CfgNode
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog

import toml
import yaml
from pydantic import BaseModel


model_mapping = {
    'faster-rcnn': 'COCO-Detection/faster_rcnn_',
    'retinanet': 'COCO-Detection/retinanet_',
    'mask-rcnn': 'COCO-InstanceSegmentation/mask_rcnn_'
}

backbone_mapping = {
    'r101-fpn': 'R_101_FPN_3x.yaml',
    'r101-c4': 'R_101_C4_3x.yaml',
    'r101-dc5': 'R_101_DC5_3x.yaml',
    'r50-fpn': 'R_50_FPN_3x.yaml',
    'r50-c4': 'R_50_C4_3x.yaml',
    'r50-dc5': 'R_50_DC5_3x.yaml'
}


class Solver(BaseModel):
    """Config for solver."""

    images_per_batch: int
    lr_policy: str
    base_lr: float
    steps: Optional[List[int]]
    max_iters: int
    checkpoint_period: Optional[int]
    eval_period: Optional[int]


class Detection(BaseModel):
    """Config for detection model training."""

    model_base: str
    override_mapping: Optional[bool] = False
    weights: Optional[str] = None
    num_classes: Optional[int]


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


class Dataloader(BaseModel):
    """Config for dataloader."""

    num_workers: int


class Config(BaseModel):
    """Overall config object."""

    detection: Detection
    solver: Solver
    dataloader: Optional[Dataloader]
    train: Optional[List[Dataset]]
    test: Optional[List[Dataset]]
    output_dir: Optional[str]


def _register(datasets: List[Dataset]) -> List[str]:
    """Register dataset in detectron2."""
    names = []
    for dataset in datasets:
        if not dataset.type == DatasetType.COCO:
            raise NotImplementedError("Currently only COCO style dataset "
                                      "structure is supported.")
        try:
            DatasetCatalog.get(dataset.name)
        except KeyError:
            register_coco_instances(dataset.name, {}, dataset.annotation_file,
                                    dataset.data_root)
            names.append(dataset.name)
        print("WARNING: You tried to register the same dataset name "
              f"twice. Skipping instance:\n{str(dataset)}")
    return names


def to_detectron2(config: Config) -> CfgNode:
    """Convert a Config object to a detectron2 readable configuration."""
    cfg = get_cfg()

    # load model base config, checkpoint
    detectron2_model_string = None
    if os.path.exists(config.detection.model_base):
        base_cfg = config.detection.model_base
    else:
        if config.detection.override_mapping:
            detectron2_model_string = config.detection.model_base
        else:
            model, backbone = config.detection.model_base.split('/')
            detectron2_model_string = model_mapping[model] + \
                                      backbone_mapping[backbone]
        base_cfg = model_zoo.get_config_file(detectron2_model_string)

    cfg.merge_from_file(base_cfg)

    # load checkpoint
    if config.detection.weights is not None:
        if os.path.exists(config.detection.weights):
            cfg.MODEL.WEIGHTS = config.detection.weights
        elif config.detection.weights == 'detectron2' and \
                detectron2_model_string is not None:
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                detectron2_model_string)
        else:
            raise ValueError(
                f'model weights path {config.detection.weights} '
                f'not found')

    # convert model attributes
    if config.detection.num_classes:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = config.detection.num_classes
        cfg.MODEL.RETINANET.NUM_CLASSES = config.detection.num_classes
    cfg.OUTPUT_DIR = config.output_dir

    # convert solver attributes
    if config.solver is not None:
        cfg.SOLVER.IMS_PER_BATCH = config.solver.images_per_batch
        cfg.SOLVER.LR_SCHEDULER_NAME = config.solver.lr_policy
        cfg.SOLVER.BASE_LR = config.solver.base_lr
        cfg.SOLVER.MAX_ITER = config.solver.max_iters
        if config.solver.checkpoint_period is not None:
            cfg.SOLVER.STEPS = config.solver.steps
        if config.solver.checkpoint_period is not None:
            cfg.SOLVER.CHECKPOINT_PERIOD = config.solver.checkpoint_period
        if config.solver.checkpoint_period is not None:
            cfg.TEST.EVAL_PERIOD = config.solver.eval_period

    # convert datalooader attributes
    if config.dataloader is not None:
        cfg.DATALOADER.NUM_WORKERS = config.dataloader.num_workers

    # register datasets
    if config.train is not None:
        cfg.DATASETS.TRAIN = _register(config.train)
    if config.test is not None:
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
        config_name = os.path.splitext(os.path.basename(filepath))[0]
        timestamp = str(datetime.now()).split('.')[0].replace(' ', '_')
        config.output_dir = os.path.join('./work_dirs/',
                                         config_name,
                                         timestamp)
    os.makedirs(config.output_dir, exist_ok=True)
    return config
