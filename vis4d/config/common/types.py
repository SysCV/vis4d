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
    """Configuration for an optimizer.

    Attributes:
        lr_scheduler (FieldConfigDictOrRef | None): Configuration for the
            learning rate scheduler. If None, no learning rate scheduler is
            used.
        lr_warmup (FieldConfigDictOrRef | None): Configuration for the
            learning rate warmup. If None, no learning rate warmup is used.
        optimizer (FieldConfigDictOrRef): Configuration for the optimizer.
        epoch_based_lr (bool | FieldReference): Whether to use epoch-based
            learning rate scheduling. If True, the learning rate scheduler is
            called at the end of each epoch. If False, the learning rate
            scheduler is called at the end of each batch.
        epoch_based_warmup (bool | FieldReference): Whether to use epoch-based
            learning rate warmup. If True, the learning rate warmup is called
            at the end of each epoch. If False, the learning rate warmup is
            called at the end of each batch.
    """

    lr_scheduler: FieldConfigDictOrRef | None
    lr_warmup: FieldConfigDictOrRef | None
    optimizer: FieldConfigDictOrRef
    epoch_based_lr: bool | FieldReference = True
    epoch_based_warmup: bool | FieldReference = True


class ExperimentParameters(FieldConfigDict):
    """Parameters for an experiment.

    Attributes:
        num_epochs (int | FieldReference): The number of epochs to train for.
    """

    num_epochs: int | FieldReference


class ExperimentConfig(FieldConfigDict):
    """Configuration for an experiment.

    This data object is used to configure an experiment. It contains the
    minimal required configuration to run an experiment. In particular, the
    data, model, optimizers, and loss need to be config dicts that can be
    instantiated as a data set, model, optimizer, and loss function,
    respectively.

    Attributes:
        data (DataConfig): Configuration for the dataset.
        output_dir (str | FieldReference): The output directory for the
            experiment.
        timestamp (str | FieldReference): The timestamp of the experiment.
        benchmark (bool | FieldReference): Whether to enable benchmarking.
        data_connector (FieldConfigDictOrRef): Configuration for the data
            connector.
        model (FieldConfigDictOrRef): Configuration for the model.
        optimizers (list[OptimizerConfig]): Configuration for the optimizers.
        loss (FieldConfigDictOrRef): Configuration for the loss function.
        callbacks (list[FieldConfigDictOrRef]): Configuration for the
            callbacks which are used in the engine.
        params (ExperimentParameters): Configuration for the experiment
            parameters.


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
