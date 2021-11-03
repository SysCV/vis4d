"""Config definitions."""
import os
from argparse import Namespace
from datetime import datetime
from typing import Any, List, Optional

import toml
import yaml
from pydantic import BaseModel, validator

from vis4d.common.utils.distributed import get_rank
from vis4d.data.datasets import BaseDatasetConfig
from vis4d.model import BaseModelConfig
from vis4d.struct import DictStrAny


class Launch(BaseModel):
    """Launch configuration.

    Standard Options (command line only):
    action (positional argument): train / test / predict
    config: Filepath to config file

    Launch Options:
    work_dir: Specific directory to save checkpoints, logs, etc. Integrates
    with exp_name and version to work_dir/exp_name/version.
    Default: ./vis4d-workspace/
    exp_name: Name of current experiment. Default: <name of model>
    version: Version of current experiment. Default: <timestamp>
    input_dir: Input directory in case you want to run inference on a folder
    with input data (e.g. images that can be temporally sorted by name).
    find_unused_parameters: Activates PyTorch checking for unused parameters
    in DDP setting. Deactivated by default for better performance.
    visualize: If you're running in predict mode, this option lets you
    visualize the model predictions in the output_dir.
    seed: Set random seed for numpy, torch, python. Default: None,
    i.e. no specific random seed is chosen.
    weights: Filepath for weights to load in test / predict. Default: "best",
    will load the best checkpoint in work_dir/exp_name/version.
    checkpoint_period: After N epochs, save out checkpoints. Default: 1
    resume: Whether to resume from weights (if specified), or last ckpt in
    work_dir/exp_name/version.
    pin_memory: Enable/Disable pin_memory option for dataloader workers in
    training.
    wandb: Use weights and biases logging instead of tensorboard (default).
    """

    action: str = ""
    samples_per_gpu: int = 1
    workers_per_gpu: int = 1
    work_dir: str = "vis4d-workspace"
    exp_name: str = ""
    version: str = (
        str(datetime.now())
        .split(".", maxsplit=1)[0]
        .replace(" ", "_")
        .replace(":", "-")
    )
    input_dir: Optional[str]
    find_unused_parameters: bool = False
    visualize: bool = False
    seed: Optional[int]
    weights: Optional[str]
    checkpoint_period: int = 1
    resume: bool = False
    pin_memory: bool = False
    wandb: bool = False

    @validator("version", always=True)
    def validate_version(  # pylint: disable=no-self-argument,no-self-use
        cls, value: str
    ) -> str:
        """Sync version in distributed setting."""
        if get_rank() == 0:
            os.environ["RUN_VERSION"] = value
        else:
            value = os.environ["RUN_VERSION"]
        return value

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


class Config(BaseModel, extra="allow"):
    """Overall config object."""

    launch: Launch = Launch()
    model: BaseModelConfig
    train: List[BaseDatasetConfig] = []
    test: List[BaseDatasetConfig] = []

    def __init__(self, **data: Any) -> None:  # type: ignore
        """Init config."""
        super().__init__(**data)
        if self.launch.exp_name == "":
            self.launch.exp_name = self.model.type


def parse_config(args: Namespace) -> Config:
    """Read config, parse cmd line arguments."""
    cfg = read_config(args.config)
    for attr, value in args.__dict__.items():
        if attr in Launch.__fields__ and value is not None:
            setattr(cfg.launch, attr, getattr(args, attr))

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
    config_dict: DictStrAny
    if ext == ".yaml":
        with open(filepath, "r", encoding="utf-8") as f:
            config_dict = yaml.load(f.read(), Loader=yaml.Loader)
    elif ext == ".toml":
        config_dict = dict(**toml.load(filepath))
    else:
        raise NotImplementedError(f"Config type {ext} not supported")
    return config_dict


def read_config(filepath: str) -> Config:
    """Read config file and parse it into Config object.

    The config file can be in yaml or toml.
    toml is recommended for readability.
    """
    config_dict = load_config(filepath)
    if "config" in config_dict:
        cwd = os.getcwd()
        os.chdir(os.path.dirname(filepath))
        subconfig_dict: DictStrAny = {}
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
        if isinstance(v, dict) and not isinstance(
            v, toml.decoder.InlineTableDict  # type: ignore
        ):
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
