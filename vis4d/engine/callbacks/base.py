"""Base module for callbacks."""
from __future__ import annotations

from torch import nn

from vis4d.common import MetricLogs
from vis4d.data.typing import DictData
from vis4d.engine.connectors import (
    SourceKeyDescription,
    get_inputs_for_pred_and_data,
    get_multi_sensor_inputs,
)

from .trainer_state import TrainerState


class Callback:
    """Base class for Callbacks."""

    def __init__(
        self,
        train_connector: None | dict[str, SourceKeyDescription] = None,
        test_connector: None | dict[str, SourceKeyDescription] = None,
        sensors: None | list[str] = None,
    ) -> None:
        """Init callback.

        Args:
            train_connector (None | dict[str, SourceKeyDescription], optional):
                Defines which which kwargs to use during training for different
                callbacks. Defaults to None.
            test_connector (None | dict[str, SourceKeyDescription], optional):
                Defines which kwargs to use during testing for different
                callbacks. Defaults to None.
            sensors (None | list[str], optional): List of sensors to use.
                Defaults to None.
        """
        self.train_connector = train_connector
        self.test_connector = test_connector
        self.sensors = sensors

    def setup(self) -> None:
        """Setup callback."""

    def get_data_connector_results(
        self, outputs: DictData, batch: DictData, train: bool
    ) -> DictData:
        """Returns the data connector results."""
        connector = self.train_connector if train else self.test_connector

        assert connector is not None, "Connector is None."
        if self.sensors is not None:
            return get_multi_sensor_inputs(
                connector, outputs, batch, self.sensors
            )

        return get_inputs_for_pred_and_data(connector, outputs, batch)

    def on_train_epoch_start(
        self, trainer_state: TrainerState, model: nn.Module
    ) -> None:
        """Hook to run at the beginning of a training epoch.

        Args:
            trainer_state (TrainerState): Trainer state.
            model (nn.Module): Model that is being trained.
        """

    def on_train_batch_end(
        self,
        trainer_state: TrainerState,
        model: nn.Module,
        outputs: DictData,
        batch: DictData,
        batch_idx: int,
    ) -> None | MetricLogs:
        """Hook to run at the end of a training batch.

        Args:
            trainer_state (TrainerState): Trainer state.
            model: Model that is being trained.
            outputs (DictData): Model prediction output.
            batch (DictData): Dataloader output data batch.
            batch_idx (int): Index of the batch.
        """

    def on_train_epoch_end(
        self, trainer_state: TrainerState, model: nn.Module
    ) -> None:
        """Hook to run at the end of a training epoch.

        Args:
            trainer_state (TrainerState): Trainer state.
            model (nn.Module): Model that is being trained.
        """

    def on_test_epoch_start(
        self, trainer_state: TrainerState, model: nn.Module
    ) -> None:
        """Hook to run at the beginning of a testing epoch.

        Args:
            trainer_state (TrainerState): Trainer state.
            model (nn.Module): Model that is being trained.
        """

    def on_test_batch_end(
        self,
        trainer_state: TrainerState,
        model: nn.Module,
        outputs: DictData,
        batch: DictData,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Hook to run at the end of a testing batch.

        Args:
            trainer_state (TrainerState): Trainer state.
            model: Model that is being trained.
            outputs (DictData): Model prediction output.
            batch (DictData): Dataloader output data batch.
            batch_idx (int): Index of the batch.
            dataloader_idx (int, optional): Index of the dataloader. Defaults
                to 0.
        """

    def on_test_epoch_end(
        self, trainer_state: TrainerState, model: nn.Module
    ) -> None | MetricLogs:
        """Hook to run at the end of a testing epoch.

        Args:
            trainer_state (TrainerState): Trainer state.
            model (nn.Module): Model that is being trained.
        """