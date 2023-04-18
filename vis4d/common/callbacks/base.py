"""Base module for callbacks."""
from __future__ import annotations

from torch import nn

from vis4d.common import DictStrAny, MetricLogs


class Callback:
    """Base class for Callbacks."""

    def __init__(
        self, run_every_nth_epoch: int = 1, num_epochs: int = -1
    ) -> None:
        """Init callback.

        Args:
            run_every_nth_epoch (int): Evaluate model every nth epoch.
                Defaults to 1.
            num_epochs (int): Number of total epochs, used for determining
                whether to evaluate at the final epoch. Defaults to -1.
        """
        self.run_every_nth_epoch = run_every_nth_epoch
        self.num_epochs = num_epochs

    def setup(self) -> None:
        """Setup callback."""

    def run_on_epoch(self, epoch: int | None) -> bool:
        """Returns whether to run callback for current epoch (default True)."""
        return (
            epoch is None
            or epoch == self.num_epochs - 1
            or epoch % self.run_every_nth_epoch == self.run_every_nth_epoch - 1
        )

    def on_train_epoch_start(self, model: nn.Module, epoch: int) -> None:
        """Hook to run at the beginning of a training epoch.

        Args:
            model (nn.Module): Model that is being trained.
            epoch (int): Current training epoch.
        """

    def on_train_batch_end(
        self,
        model: nn.Module,
        shared_inputs: DictStrAny,
        inputs: DictStrAny,
    ) -> None:
        """Hook to run at the end of a training batch.

        Args:
            model: Model that is being trained.
            shared_inputs (ArgsType): Shared inputs for callback.
            inputs (ArgsType): Inputs for callback.
        """

    def on_train_epoch_end(self, model: nn.Module, epoch: int) -> None:
        """Hook to run at the end of a training epoch.

        Args:
            model (nn.Module): Model that is being trained.
            epoch (int): Current training epoch.
        """

    def on_test_epoch_start(self, model: nn.Module, epoch: int) -> None:
        """Hook to run at the beginning of a testing epoch.

        Args:
            model (nn.Module): Model that is being trained.
            epoch (int): Current testing epoch.
        """

    def on_test_batch_end(
        self, model: nn.Module, shared_inputs: DictStrAny, inputs: DictStrAny
    ) -> None:
        """Hook to run at the end of a testing batch.

        Args:
            model: Model that is being trained.
            shared_inputs (ArgsType): Shared inputs for callback.
            inputs (ArgsType): Inputs for callback.
        """

    def on_test_epoch_end(
        self, model: nn.Module, epoch: None | int = None
    ) -> None | MetricLogs:
        """Hook to run at the end of a testing epoch.

        Args:
            model (nn.Module): Model that is being trained.
            epoch (int): Current testing epoch.
        """
