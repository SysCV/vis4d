"""Config definitions."""
import os
from argparse import Namespace
from datetime import datetime
from typing import Any, List, Optional

import toml
import yaml
from pydantic import BaseModel, validator

from vis4d.common.utils.distributed import get_rank
from vis4d.struct import DictStrAny, ModuleCfg


class Launch(BaseModel):
    """Launch configuration.

    Standard Options (command line only):
    action (positional argument): train / test / predict
    config: Filepath to config file

    Launch Options:
    """

    action: str = ""


class Config(BaseModel, extra="allow"):
    """Overall config object."""

    launch: Launch = Launch()


def parse_config(args: Namespace) -> Config:
    """Read config, parse cmd line arguments."""
    cfg = read_config(args.config)
    for attr, value in args.__dict__.items():
        if attr in Launch.__fields__ and value is not None:
            if (
                not isinstance(value, bool)
                or not value == Launch().__dict__[attr]
            ):
                setattr(cfg.launch, attr, getattr(args, attr))
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
    if filepath == "":  # TODO make pretty
        return Config()
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
            v, toml.decoder.InlineTableDict
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
