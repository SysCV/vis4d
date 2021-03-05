"""Config definitions."""

from typing import List

from pydantic import BaseModel
import yaml
import toml

from os.path import splitext


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


class Config(BaseModel):
    """Overall config object."""

    detection: Detection
    solver: Solver


def read_config(filepath: str) -> Config:
    """Read config file and parse it into Config object.
    The config file can be in yaml or toml.
    toml is recommended for readability.
    """
    ext = splitext(filepath)[1]
    if ext == ".yaml":
        config_dict = yaml.load(
            open(filepath, "r").read(),
            Loader=yaml.Loader,
        )
    elif ext == ".toml":
        config_dict = toml.load(filepath)
    config = Config(**config_dict)
    return config
