"""LightningModule that wraps around the vis4d models, losses and optims."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytorch_lightning as pl
from torch import nn, optim

from vis4d.data.typing import DictData
from vis4d.engine.connectors import DataConnector
from vis4d.engine.opt import Optimizer


class TorchOptimizer(optim.Optimizer):
    """Wrapper around vis4d optimizer to make it compatible with pl."""

    def __init__(self, optimizer: Optimizer, model: nn.Module) -> None:
        """Creates a new Optimizer.

        Args:
            optimizer: The vis4d optimizer to wrap.
            model: The model to optimize.
        """
        self.optim = optimizer
        self.optim.setup(model)
        self._step = 0
        super().__init__(
            params=self.optim.optimizer.param_groups,
            defaults=self.optim.optimizer.defaults,
        )

    def step(self, closure: Callable[[], float] | None = None) -> None:
        """Performs a single optimization step.

        Args:
           closure: A closure that reevaluates the model and returns the loss.
        """
        self.optim.step(self._step, closure)
        self._step += 1

    def zero_grad(self) -> None:  # pylint: disable=arguments-differ
        """Clears the gradients of all optimized parameters."""
        self.optim.zero_grad()


class TrainingModule(pl.LightningModule):  # pylint: disable=too-many-ancestors
    """LightningModule that wraps around the vis4d implementations.

    This is a wrapper around the vis4d implementations that allows to use
    pytorch-lightning for training and testing.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizers: list[Optimizer],
        loss: nn.Module,
        data_connector: DataConnector,
    ):
        """Initialize the TrainingModule.

        Args:
            model: The model to train.
            optimizers: The optimizers to use. Will be wrapped into a pytorch
                optimizer.
            loss: The loss function to use.
            data_connector: The data connector to use.
        """
        super().__init__()
        self.model = model
        self.optims = optimizers
        self.loss = loss
        self.data_connector = data_connector

    def forward(  # pylint: disable=arguments-differ,line-too-long,unused-argument # type: ignore
        self, data: DictData
    ) -> Any:
        """Forward pass through the model."""
        return self.model(**self.data_connector.get_train_input(data))

    def training_step(  # pylint: disable=arguments-differ,line-too-long,unused-argument # type: ignore
        self, batch: DictData, batch_idx: int
    ) -> Any:
        """Perform a single training step."""
        out = self.model(**self.data_connector.get_train_input(batch))
        l = self.loss(**self.data_connector.get_loss_input(out, batch))
        return {"loss": sum(l.values()), "predictions": out}

    def validation_step(  # pylint: disable=arguments-differ,line-too-long,unused-argument # type: ignore
        self, batch: DictData, batch_idx: int, dataloader_idx: int = 0
    ) -> Any:
        """Perform a single validation step."""
        out = self.model(**self.data_connector.get_test_input(batch))
        return out

    def test_step(  # pylint: disable=arguments-differ,line-too-long,unused-argument # type: ignore
        self, batch: DictData, batch_idx: int, dataloader_idx: int = 0
    ) -> Any:
        """Perform a single test step."""
        out = self.model(**self.data_connector.get_test_input(batch))
        return out

    def configure_optimizers(self) -> list[TorchOptimizer]:
        """Return the optimizer to use."""
        return [TorchOptimizer(o, self.model) for o in self.optims]
