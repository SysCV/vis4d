"""Config definitions."""
import os
import sys
from argparse import Namespace
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import toml
import yaml
from pydantic import BaseModel, validator

from openmt.data import DataloaderConfig as Dataloader
from openmt.model import BaseModelConfig


class Solver(BaseModel):
    """Config for solver."""

    images_per_batch: int
    lr_policy: str
    base_lr: float
    steps: Optional[List[int]]
    max_iters: int
    checkpoint_period: Optional[int]
    eval_period: Optional[int]


class DatasetType(str, Enum):
    """Enum for dataset type.

    coco: COCO style dataset to support detectron2 training.
    scalabel_video: Scalabel based video dataset format.
    custom: Custom dataset type for user-defined datasets.
    """

    COCO = "coco"
    SCALABEL_VIDEO = "scalabel_video"
    CUSTOM = "custom"


class Dataset(BaseModel):
    """Config for training/evaluation datasets."""

    name: str
    type: DatasetType
    data_root: str
    annotations: str


class Launch(BaseModel):
    """Launch configuration.

    Standard Options (command line only):
    action (positional argument): train / predict routine
    config: Filepath to config file

    Launch Options:
    device: Device to train on (cpu / cuda / ..)
    weights: Filepath for weights to load. Set to "detectron2" If you want to
            load weights from detectron2 for a corresponding detector.
    num_gpus:"number of gpus *per machine*"
    num_machines: "total number of machines"
    machine_rank: the rank of this machine (unique per machine)
    dist_url: initialization URL for pytorch distributed backend. See
        https://pytorch.org/docs/stable/distributed.html for details.
    resume: Whether to attempt to resume from the checkpoint directory.
    eval_only: perform evaluation only
    """

    device: str = "cpu"
    weights: Optional[str] = None
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

    model: BaseModelConfig
    solver: Solver
    dataloader: Dataloader
    train: Optional[List[Dataset]]
    test: Optional[List[Dataset]]
    output_dir: Optional[str]
    launch: Launch = Launch()

    @validator("output_dir", always=True)
    def validate_output_dir(  # type: ignore # pylint: disable=no-self-argument,no-self-use,line-too-long
        cls, value: str, values: Dict[str, Any]
    ) -> str:
        """Check if output dir, create output dir if necessary."""
        if value is None:
            timestamp = str(datetime.now()).split(".")[0].replace(" ", "_")
            value = os.path.join(
                "openmt-workspace", values["model"].type, timestamp
            )
        os.makedirs(value, exist_ok=True)
        return value


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
    return config
