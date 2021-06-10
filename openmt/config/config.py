"""Config definitions."""
import os
import sys
from argparse import Namespace
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, no_type_check

import toml
import yaml
from pydantic import BaseModel, validator

from openmt.common.io import DataBackendConfig
from openmt.model import BaseModelConfig


class ReferenceSamplingConfig(BaseModel):
    """Config for customizing the sampling of reference views."""

    type: str = "uniform"
    num_ref_imgs: int = 0
    scope: int = 1
    frame_order: str = "key_first"
    skip_nomatch_samples: bool = False

    @validator("scope")
    def validate_scope(  # type: ignore # pylint: disable=no-self-argument,no-self-use, line-too-long
        cls, value: int, values
    ) -> int:
        """Check scope attribute."""
        if not value > values["num_ref_imgs"] // 2:
            raise ValueError("Scope must be higher than num_ref_imgs / 2.")
        return value


class Augmentation(BaseModel):
    """Data augmentation instance config."""

    type: str
    kwargs: Dict[str, Union[bool, float, str, Tuple[int, int]]]


class DataloaderConfig(BaseModel):
    """Config for dataloader."""

    data_backend: DataBackendConfig = DataBackendConfig()
    workers_per_gpu: int
    inference_sampling: str = "sample_based"
    categories: Optional[List[str]] = None
    skip_empty_samples: bool = False
    compute_global_instance_ids: bool = False
    train_augmentations: Optional[List[Augmentation]] = None
    test_augmentations: Optional[List[Augmentation]] = None
    ref_sampling_cfg: ReferenceSamplingConfig

    @validator("inference_sampling", check_fields=False)
    def validate_inference_sampling(  # pylint: disable=no-self-argument,no-self-use,line-too-long
        cls, value: str
    ) -> str:
        """Check inference_sampling attribute."""
        if value not in ["sample_based", "sequence_based"]:
            raise ValueError(
                "inference_sampling must be sample_based or sequence_based"
            )
        return value


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
    eval_metrics: List[str]


class Dataset(BaseModel):
    """Config for training/evaluation datasets."""

    name: str
    type: str
    data_root: str
    annotations: Optional[str]
    config_path: Optional[str]


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
    train: Optional[List[Dataset]]
    test: Optional[List[Dataset]]
    launch: Launch = Launch()

    def __init__(self, **data: Any) -> None:  # type: ignore
        """Init config."""
        super().__init__(**data)
        if self.launch.output_dir == "":
            timestamp = str(datetime.now()).split(".")[0].replace(" ", "_")
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

    if args.__dict__.get("cfg_options", "") != "":
        cfg_dict = cfg.dict()
        options = args.cfg_options.split(",")

        @no_type_check
        def update(my_dict, key_list, value):
            cur_key = key_list.pop(0)
            if len(key_list) == 0:
                my_dict[cur_key] = value
                return
            update(my_dict[cur_key], key_list, value)

        for option in options:
            key, value = option.split("=")
            update(cfg_dict, key.split("."), value)
        cfg = Config(**cfg_dict)

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

    # Fix pickle error with the class not being serializable:
    # TomlDecoder.get_empty_inline_table.<locals>.DynamicInlineTableDict
    @no_type_check
    def check_for_dicts(obj):
        if isinstance(obj, dict):
            return {k: check_for_dicts(v) for k, v in obj.items()}
        return obj

    config_dict = check_for_dicts(config_dict)

    config = Config(**config_dict)
    return config
