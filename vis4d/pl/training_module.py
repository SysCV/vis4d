"""LightningModule that wraps around the vis4d models, losses and optims."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytorch_lightning as pl
import torch
from torch import nn, optim
from torchmetrics import MeanMetric

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
        assert self.optim.optimizer is not None
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
        self.optim.step_on_batch(self._step, closure)
        self._step += 1

    def step_on_epoch(self, epoch: int) -> None:
        """Performs a single optimization step.

        Args:
           epoch (int): The current epoch of the training loop.
        """
        self.optim.step_on_epoch(epoch)

    def zero_grad(self, set_to_none: bool = False) -> None:
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
        self.loss_fn = loss
        self.data_connector = data_connector

    def forward(  # type: ignore # pylint: disable=arguments-differ,line-too-long,unused-argument
        self, data: DictData
    ) -> Any:
        """Forward pass through the model."""
        return self.model(**self.data_connector.get_train_input(data))

    def training_step(  # type: ignore # pylint: disable=arguments-differ,line-too-long,unused-argument
        self, batch: DictData, batch_idx: int
    ) -> Any:
        """Perform a single training step."""
        out = self.model(**self.data_connector.get_train_input(batch))
        losses = self.loss_fn(**self.data_connector.get_loss_input(out, batch))
        if isinstance(losses, torch.Tensor):
            losses = {"loss": losses}
        else:
            losses["loss"] = sum(list(losses.values()))

        log_dict = {}
        metric_attributes = []
        for k, v in losses.items():
            if not hasattr(self, k):
                metric = MeanMetric()
                metric.to(self.device)  # type: ignore
                setattr(self, k, metric)

            metric = getattr(self, k)
            metric(v.detach())
            log_dict["train/" + k] = metric
            metric_attributes += [k]

        for (k, v), k_name in zip(log_dict.items(), metric_attributes):
            self.log(
                k,
                v,
                logger=True,
                prog_bar=True,
                on_step=True,
                on_epoch=False,
                metric_attribute=k_name,
            )
        return {
            "loss": losses["loss"],
            "metrics": losses,
            "predictions": out,
        }

    def validation_step(  # type: ignore  # pylint: disable=arguments-differ,line-too-long,unused-argument
        self, batch: DictData, batch_idx: int, dataloader_idx: int = 0
    ) -> Any:
        """Perform a single validation step."""
        out = self.model(**self.data_connector.get_test_input(batch))
        return out

    def test_step(  # type: ignore  # pylint: disable=arguments-differ,line-too-long,unused-argument
        self, batch: DictData, batch_idx: int, dataloader_idx: int = 0
    ) -> Any:
        """Perform a single test step."""
        out = self.model(**self.data_connector.get_test_input(batch))
        return out

    def configure_optimizers(self) -> list[TorchOptimizer]:
        """Return the optimizer to use."""
        return [TorchOptimizer(o, self.model) for o in self.optims]
