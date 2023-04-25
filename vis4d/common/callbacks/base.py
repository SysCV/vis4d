"""Base module for callbacks."""
from __future__ import annotations

from torch import nn

from vis4d.common import MetricLogs, TrainerType
from vis4d.data.typing import DictData
from vis4d.engine.connectors import SourceKeyDescription


class Callback:
    """Base class for Callbacks."""

    def __init__(
        self,
        every_n_epochs: int = 1,
        run_last: bool = True,
        num_epochs: int = -1,
        connector: None | dict[str, dict[str, SourceKeyDescription]] = None,
    ) -> None:
        """Init callback.

        Args:
            run_every_nth_epoch (int): Evaluate model every nth epoch.
                Defaults to 1.
            num_epochs (int): Number of total epochs, used for determining
                whether to evaluate at the final epoch. Defaults to -1.
            connector (None | dict[str, dict[str, SourceKeyDescription]],
                optional): Defines which kwargs to use for different callbacks.
        """
        self.every_n_epochs = every_n_epochs
        self.run_last = run_last
        self.num_epochs = num_epochs
        self.connector = connector

    def setup(self) -> None:
        """Setup callback."""

    def run_on_epoch(self, epoch: int) -> bool:
        """Returns whether to run callback for current epoch."""
        if epoch == self.num_epochs - 1 and self.run_last:
            return True
        elif epoch % self.every_n_epochs == self.every_n_epochs - 1:
            return True
        return False

    def on_train_epoch_start(
        self, trainer: TrainerType, model: nn.Module
    ) -> None:
        """Hook to run at the beginning of a training epoch.

        Args:
            callback_inputs (CallbackInputs): General inputs to callback.
            model (nn.Module): Model that is being trained.
        """

    def on_train_batch_end(
        self,
        trainer: TrainerType,
        model: nn.Module,
        outputs: DictData,
        batch: DictData,
        batch_idx: int,
    ) -> None:
        """Hook to run at the end of a training batch.

        Args:
            callback_inputs (CallbackInputs): General inputs to callback.
            model: Model that is being trained.
            predictions (DictData): Model prediction output.
            data (DictData): Dataloader output.
        """

    def on_train_epoch_end(
        self, trainer: TrainerType, model: nn.Module
    ) -> None:
        """Hook to run at the end of a training epoch.

        Args:
            callback_inputs (CallbackInputs): General inputs to callback.
            model (nn.Module): Model that is being trained.
        """

    def on_test_epoch_start(
        self, trainer: TrainerType, model: nn.Module
    ) -> None:
        """Hook to run at the beginning of a testing epoch.

        Args:
            callback_inputs (CallbackInputs): General inputs to callback.
            model (nn.Module): Model that is being trained.
        """

    def on_test_batch_end(
        self,
        trainer: TrainerType,
        model: nn.Module,
        outputs: DictData,
        batch: DictData,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Hook to run at the end of a testing batch.

        Args:
            callback_inputs (CallbackInputs): General inputs to callback.
            model: Model that is being trained.
            predictions (DictData): Model prediction output.
            data (DictData): Dataloader output.
        """

    def on_test_epoch_end(
        self, trainer: TrainerType, model: nn.Module
    ) -> None | MetricLogs:
        """Hook to run at the end of a testing epoch.

        Args:
            callback_inputs (CallbackInputs): General inputs to callback.
            model (nn.Module): Model that is being trained.
        """
