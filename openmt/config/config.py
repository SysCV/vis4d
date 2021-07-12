"""Config definitions."""
import os
import sys
from argparse import Namespace
from datetime import datetime
from typing import Any, List, Optional

import toml
import yaml
from pydantic import BaseModel, validator

from openmt.data.dataset_mapper import DataloaderConfig
from openmt.data.datasets.base import BaseDatasetConfig
from openmt.model import BaseModelConfig
from openmt.struct import DictStrAny


class Solver(BaseModel):
    """Config for solver."""

    images_per_gpu: int
    lr_policy: str
    base_lr: float
    steps: Optional[List[int]]
    max_iters: int
    warmup_iters: Optional[int]
    checkpoint_period: Optional[int]
    log_period: Optional[int]
    eval_period: Optional[int]


class Launch(BaseModel):
    """Launch configuration.

    Standard Options (command line only):
    action (positional argument): train / predict routine
    config: Filepath to config file

    Launch Options:
    device: Device to train on (cpu / cuda / ..)
    weights: Filepath for weights to load. Set to "detectron2" If you want to
            load weights from detectron2 for a corresponding detector.
    num_gpus: number of gpus per machine
    num_machines: total number of machines
    machine_rank: the rank of this machine (unique per machine)
    dist_url: initialization URL for pytorch distributed backend. See
        https://pytorch.org/docs/stable/distributed.html for details.
    resume: Whether to attempt to resume from the checkpoint directory.
    input_dir: Input directory in case you want to run inference on a folder
    with input data (e.g. images that can be temporally sorted by name)
    output_dir: Specific directory to save checkpoints, logs, etc.
    Default: openmt-workspace/<model_name>/<timestamp>
    visualize: If you're running in predict mode, this option lets you
    visualize the model predictions in the output_dir.
    seed: Set random seed for numpy, torch, python. Default: -1,
    i.e. no specific random seed is chosen.
    """

    action: str = ""
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
    input_dir: Optional[str]
    output_dir: str = ""
    visualize: bool = False
    seed: int = -1

    @validator("input_dir", always=True)
    def validate_input_dir(  # pylint: disable=no-self-argument,no-self-use
        cls, value: Optional[str]
    ) -> Optional[str]:
        """Check if input dir exists."""
        if value is not None:
            if not os.path.exists(value):
                raise FileNotFoundError(
                    f"Input directory does not exist: {value}"
                )
        return value


class Config(BaseModel):
    """Overall config object."""

    model: BaseModelConfig
    solver: Solver
    dataloader: DataloaderConfig
    train: List[BaseDatasetConfig] = []
    test: List[BaseDatasetConfig] = []
    launch: Launch = Launch()

    def __init__(self, **data: Any) -> None:  # type: ignore
        """Init config."""
        super().__init__(**data)
        if self.launch.output_dir == "":
            timestamp = (
                str(datetime.now())
                .split(".", maxsplit=1)[0]
                .replace(" ", "_")
                .replace(":", "-")
            )
            self.launch.output_dir = os.path.join(
                "openmt-workspace", self.model.type, timestamp
            )
        os.makedirs(self.launch.output_dir, exist_ok=True)


def parse_config(args: Namespace) -> Config:
    """Read config, parse cmd line arguments, create workspace dir."""
    cfg = read_config(args.config)
    for attr, value in args.__dict__.items():
        if attr in Launch.__fields__ and value is not None:
            setattr(cfg.launch, attr, getattr(args, attr))

    if cfg.launch.device == "cpu":
        cfg.launch.num_gpus = 0

    if args.__dict__.get("cfg_options", "") != "":
        cfg_dict = cfg.dict()
        options = args.cfg_options.split(",")
        for option in options:
            key, value = option.split("=")
            keylist_update(cfg_dict, key.split("."), value)
        cfg = Config(**cfg_dict)
    return cfg


def load_config(filepath: str) -> DictStrAny:
    """Load config from file to dict."""
    ext = os.path.splitext(filepath)[1]
    if ext == ".yaml":
        with open(filepath, "r") as f:
            config_dict = yaml.load(f.read(), Loader=yaml.Loader)
    elif ext == ".toml":
        config_dict = toml.load(filepath)
    else:
        raise NotImplementedError(f"Config type {ext} not supported")
    return config_dict  # type: ignore


def read_config(filepath: str) -> Config:
    """Read config file and parse it into Config object.

    The config file can be in yaml or toml.
    toml is recommended for readability.
    """
    config_dict = load_config(filepath)
    if "config" in config_dict:
        cwd = os.getcwd()
        os.chdir(os.path.dirname(filepath))
        subconfig_dict = dict()  # type: DictStrAny
        for cfg in config_dict["config"]:
            assert "path" in cfg, "Config arguments must have path!"
            nested_update(subconfig_dict, load_config(cfg["path"]))

        nested_update(subconfig_dict, config_dict)
        config_dict = subconfig_dict
        os.chdir(cwd)

    config_dict = check_for_dicts(config_dict)
    return Config(**config_dict)


def keylist_update(  # type: ignore
    my_dict: DictStrAny, key_list: List[str], value: Any
) -> None:
    """Update nested dict based on multiple keys saved in a list."""
    cur_key = key_list.pop(0)
    if len(key_list) == 0:
        my_dict[cur_key] = value
        return
    keylist_update(my_dict[cur_key], key_list, value)


def nested_update(ori: DictStrAny, new: DictStrAny) -> DictStrAny:
    """Update function for updating a nested dict."""
    for k, v in new.items():
        if isinstance(v, dict):
            ori[k] = nested_update(ori.get(k, {}), v)
        else:
            ori[k] = v
    return ori


def check_for_dicts(obj: Any) -> Any:  # type: ignore
    """Fix pickle error with a class not being serializable.

    TomlDecoder.get_empty_inline_table.<locals>.DynamicInlineTableDict
    """
    if isinstance(obj, dict):
        return {k: check_for_dicts(v) for k, v in obj.items()}
    return obj
