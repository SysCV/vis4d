"""Type definitions for configuration files."""
from __future__ import annotations

from typing import Any, Union

from ml_collections import FieldReference

from vis4d.config.config_dict import FieldConfigDict
from vis4d.engine.optim import ParamGroupsCfg

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


class LrSchedulerConfig(FieldConfigDict):
    """Configuration for a learning rate scheduler.

    Attributes:
        scheduler (FieldConfigDict): Configuration for the learning rate
            scheduler.
        begin (int): Begin epoch.
        end (int): End epoch.
        epoch_based (bool): Whether the learning rate scheduler is epoch based
            or step based.
    """

    scheduler: FieldConfigDict
    begin: int
    end: int
    epoch_based: bool


class OptimizerConfig(FieldConfigDict):
    """Configuration for an optimizer.

    Attributes:
        optimizer (FieldConfigDictOrRef): Configuration for the optimizer.
        lr_scheduler (list[LrSchedulerConfig] | None): Configuration for the
            learning rate scheduler.
        param_groups (list[ParamGroupsCfg] | None): Configuration for the
            parameter groups.
    """

    optimizer: FieldConfigDictOrRef
    lr_scheduler: list[LrSchedulerConfig] | None
    param_groups: list[ParamGroupsCfg] | None


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
        work_dir (str | FieldReference): The working directory for the
            experiment.
        experiment_name (str | FieldReference): The name of the experiment.
        timestamp (str | FieldReference): The timestamp of the experiment.
        output_dir (str | FieldReference): The output directory for the
            experiment.
        seed (int | FieldReference): The random seed for the experiment.
        log_every_n_steps (int | FieldReference): The number of steps after
            which the logs should be written.
        use_tf32 (bool | FieldReference): Whether to use tf32.
        benchmark (bool | FieldReference): Whether to enable benchmarking.
        data_connector (FieldConfigDictOrRef): Configuration for the data
            connector.
        model (FieldConfigDictOrRef): Configuration for the model.
        loss (FieldConfigDictOrRef): Configuration for the loss function.
        optimizers (list[OptimizerConfig]): Configuration for the optimizers.
        callbacks (list[FieldConfigDictOrRef]): Configuration for the
            callbacks which are used in the engine.
        params (ExperimentParameters): Configuration for the experiment
            parameters.
    """

    # Data
    data: DataConfig

    # Base
    work_dir: str | FieldReference
    experiment_name: str | FieldReference
    timestamp: str | FieldReference
    output_dir: str | FieldReference
    seed: int | FieldReference
    log_every_n_steps: int | FieldReference
    use_tf32: bool | FieldReference
    benchmark: bool | FieldReference

    # Data connector
    data_connector: FieldConfigDictOrRef

    # Model
    model: FieldConfigDictOrRef

    # Loss
    loss: FieldConfigDictOrRef

    # Optimizer
    optimizers: list[OptimizerConfig]

    # Callbacks
    callbacks: list[FieldConfigDictOrRef] = []

    params: ExperimentParameters


class ParameterSweepConfig(FieldConfigDict):
    """Configuration for a parameter sweep.

    Confguration object for a parameter sweep. It contains the minimal required
    configuration to run a parameter sweep.

    Attributes:
        method (str): Sweep method that should be used (e.g. grid)
        sampling_args (list[tuple[str, Any]]): Arguments that should be passed
            to the sweep method. E.g. for grid, this would be a list of tuples
            of the form (parameter_name, parameter_values).
        suffix (str): Suffix that should be appended to the output directory.
            This will be interpreted as a string template and can contain
            references to the sampling_args.
            E.g. "lr_{lr:.2e}_bs_{batch_size}".
    """

    method: str | FieldReference
    sampling_args: list[tuple[str, Any]] | FieldReference  # type: ignore
    suffix: str | FieldReference = ""
