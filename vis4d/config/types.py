"""Type definitions for configuration files."""
from __future__ import annotations

from typing import Union

from ml_collections import FieldReference

from vis4d.config.util import ConfigDict

ConfigDictOrRef = Union[ConfigDict, FieldReference]


class DataConfig(ConfigDict):
    """Configuration for a data set.

    This data object is used to configure the training and test data of an
    experiment. In particular, the train_dataloader and test_dataloader
    need to be config dicts that can be instantiated as a dataloader.

    Attributes:
        train_dataloader (ConfigDict): Configuration for the training
           dataloader.
        test_dataloader (ConfigDict): Configuration for the test dataloader.


    Example:
        >>> from vis4d.config.types import DataConfig
        >>> from vis4d.config.util import class_config
        >>> from my_package.data import MyDataLoader
        >>> cfg = DataConfig()
        >>> cfg.train_dataloader = class_config(MyDataLoader, ...)
    """

    train_dataloader: ConfigDict
    test_dataloader: ConfigDict


class OptimizerConfig(ConfigDict):
    """Configuration for an optimizer."""

    lr_scheduler: ConfigDictOrRef
    lr_warmup: ConfigDictOrRef
    optimizer: ConfigDictOrRef
    epoch_based_lr: bool | FieldReference
    epoch_based_warmup: bool | FieldReference


class ExperimentParameters(ConfigDict):
    """Parameters for an experiment."""

    num_epochs: int | FieldReference


class ExperimentConfig(ConfigDict):
    """Configuration for an experiment.

    This data object is used to configure an experiment. It contains the
    minimal required configuration to run an experiment. In particular, the
    data, model, optimizers, and loss need to be config dicts that can be
    instantiated as a data set, model, optimizer, and loss function,
    respectively.
    """

    # Experiment description
    data: DataConfig

    output_dir: str | FieldReference
    timestamp: str | FieldReference
    benchmark: bool | FieldReference
    data_connector: ConfigDictOrRef

    model: ConfigDictOrRef
    optimizers: list[OptimizerConfig]
    loss: ConfigDictOrRef

    shared_callbacks: dict[str, ConfigDictOrRef] | None
    train_callbacks: dict[str, ConfigDictOrRef] | None
    test_callbacks: dict[str, ConfigDictOrRef] | None

    params: ExperimentParameters
