"""Config definitions."""
import os
import sys
from argparse import Namespace
from datetime import datetime
from enum import Enum
from typing import List, Optional

import toml
import yaml
from pydantic import BaseModel


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
    device: Optional[str]


class DatasetType(str, Enum):
    """Enum for dataset type.

    coco: COCO style dataset to support detectron2 training.
    custom: Custom dataset type for user-defined datasets.
    """

    COCO = "coco"
    CUSTOM = "custom"


class Dataset(BaseModel):
    """Config for training/evaluation datasets."""

    name: str
    type: DatasetType
    data_root: str
    annotation_file: Optional[str]


class Dataloader(BaseModel):
    """Config for dataloader."""

    num_workers: int


class Launch(BaseModel):
    """Launch configuration."""

    num_gpus: int = 1
    num_machines: int = 1
    machine_rank: int = 0
    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = (
        2 ** 15
        + 2 ** 14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )
    dist_url: str = "tcp://127.0.0.1:{}".format(port)
    resume: bool = False
    eval_only: bool = False


class Config(BaseModel):
    """Overall config object."""

    detection: Detection
    solver: Solver
    dataloader: Optional[Dataloader]
    train: Optional[List[Dataset]]
    test: Optional[List[Dataset]]
    output_dir: Optional[str]
    launch: Launch = Launch()


def parse_config(args: Namespace) -> Config:
    """Read config, parse cmd line arguments."""
    cfg = read_config(args.config)

    for attr, value in args.__dict__.items():
        if attr in Launch.__fields__ and value is not None:
            setattr(cfg.launch, attr, getattr(args, attr))

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
        timestamp = str(datetime.now()).split(".")[0].replace(" ", "_")
        config.output_dir = os.path.join(
            "./work_dirs/", config_name, timestamp
        )
    os.makedirs(config.output_dir, exist_ok=True)

    return config
