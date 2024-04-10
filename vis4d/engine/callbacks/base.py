"""Base module for callbacks."""

from __future__ import annotations

from torch import Tensor, nn

from vis4d.common.typing import DictStrArrNested, MetricLogs
from vis4d.data.typing import DictData
from vis4d.engine.connectors import CallbackConnector
from vis4d.engine.loss_module import LossModule

from .trainer_state import TrainerState


class Callback:
    """Base class for Callbacks."""

    def __init__(
        self,
        epoch_based: bool = True,
        train_connector: None | CallbackConnector = None,
        test_connector: None | CallbackConnector = None,
    ) -> None:
        """Init callback.

        Args:
            epoch_based (bool, optional): Whether the callback is epoch based.
                Defaults to False.
            train_connector (None | CallbackConnector, optional): Defines which
                kwargs to use during training for different callbacks. Defaults
                to None.
            test_connector (None | CallbackConnector, optional): Defines which
                kwargs to use during testing for different callbacks. Defaults
                to None.
        """
        self.epoch_based = epoch_based
        self.train_connector = train_connector
        self.test_connector = test_connector

    def setup(self) -> None:
        """Setup callback."""

    def get_train_callback_inputs(
        self, outputs: DictData, batch: DictData
    ) -> dict[str, Tensor | DictStrArrNested]:
        """Returns the data connector results for training.

        It extracts the required data from prediction and datas and passes it
        to the next component with the provided new key.

        Args:
            outputs (DictData): Outputs of the model.
            batch (DictData): Batch data.

        Returns:
            dict[str, Tensor | DictStrArrNested]: Data connector results.

        Raises:
            AssertionError: If train connector is None.
        """
        assert self.train_connector is not None, "Train connector is None."

        return self.train_connector(outputs, batch)

    def get_test_callback_inputs(
        self, outputs: DictData, batch: DictData
    ) -> dict[str, Tensor | DictStrArrNested]:
        """Returns the data connector results for inference.

        It extracts the required data from prediction and datas and passes it
        to the next component with the provided new key.

        Args:
            outputs (DictData): Outputs of the model.
            batch (DictData): Batch data.

        Returns:
            dict[str, Tensor | DictStrArrNested]: Data connector results.

        Raises:
            AssertionError: If test connector is None.
        """
        assert self.test_connector is not None, "Test connector is None."

        return self.test_connector(outputs, batch)

    def on_train_batch_start(
        self,
        trainer_state: TrainerState,
        model: nn.Module,
        loss_module: LossModule,
        batch: DictData,
        batch_idx: int,
    ) -> None:
        """Hook to run at the start of a training batch.

        Args:
            trainer_state (TrainerState): Trainer state.
            model: Model that is being trained.
            loss_module (LossModule): Loss module.
            batch (DictData): Dataloader output data batch.
            batch_idx (int): Index of the batch.
        """

    def on_train_epoch_start(
        self,
        trainer_state: TrainerState,
        model: nn.Module,
        loss_module: LossModule,
    ) -> None:
        """Hook to run at the beginning of a training epoch.

        Args:
            trainer_state (TrainerState): Trainer state.
            model (nn.Module): Model that is being trained.
            loss_module (LossModule): Loss module.
        """

    def on_train_batch_end(
        self,
        trainer_state: TrainerState,
        model: nn.Module,
        loss_module: LossModule,
        outputs: DictData,
        batch: DictData,
        batch_idx: int,
    ) -> None | MetricLogs:
        """Hook to run at the end of a training batch.

        Args:
            trainer_state (TrainerState): Trainer state.
            model: Model that is being trained.
            loss_module (LossModule): Loss module.
            outputs (DictData): Model prediction output.
            batch (DictData): Dataloader output data batch.
            batch_idx (int): Index of the batch.
        """

    def on_train_epoch_end(
        self,
        trainer_state: TrainerState,
        model: nn.Module,
        loss_module: LossModule,
    ) -> None:
        """Hook to run at the end of a training epoch.

        Args:
            trainer_state (TrainerState): Trainer state.
            model (nn.Module): Model that is being trained.
            loss_module (LossModule): Loss module.
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
