"""Type definitions for configuration files."""
from __future__ import annotations

from typing import Union

from ml_collections import FieldReference

from vis4d.config.config_dict import FieldConfigDict

FieldConfigDictOrRef = Union[FieldConfigDict, FieldReference]


class DataConfig(FieldConfigDict):
    """Configuration for a data set.

    This data object is used to configure the training and test data of an
    experiment. In particular, the train_dataloader and test_dataloader
    need to be config dicts that can be instantiated as a dataloader.

    Attributes:
        train_dataloader (FieldConfigDict): Configuration for the training
           dataloader.
        test_dataloader (FieldConfigDict): Configuration for the test
            dataloader.


    Example:
        >>> from vis4d.config.types import DataConfig
        >>> from vis4d.config.util import class_config
        >>> from my_package.data import MyDataLoader
        >>> cfg = DataConfig()
        >>> cfg.train_dataloader = class_config(MyDataLoader, ...)
    """

    train_dataloader: FieldConfigDict
    test_dataloader: FieldConfigDict


class OptimizerConfig(FieldConfigDict):
    """Configuration for an optimizer."""

    lr_scheduler: FieldConfigDictOrRef | None
    lr_warmup: FieldConfigDictOrRef | None
    optimizer: FieldConfigDictOrRef
    epoch_based_lr: bool | FieldReference = True
    epoch_based_warmup: bool | FieldReference = True


class ExperimentParameters(FieldConfigDict):
    """Parameters for an experiment."""

    num_epochs: int | FieldReference


class ExperimentConfig(FieldConfigDict):
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
    data_connector: FieldConfigDictOrRef

    model: FieldConfigDictOrRef
    optimizers: list[OptimizerConfig]
    loss: FieldConfigDictOrRef

    callbacks: list[FieldConfigDictOrRef] = []

    params: ExperimentParameters
