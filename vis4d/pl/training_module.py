"""LightningModule that wraps around the models, losses and optims."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.core.optimizer import LightningOptimizer
from ml_collections import ConfigDict
from torch import Tensor, optim

from vis4d.common.distributed import broadcast
from vis4d.common.logging import rank_zero_info
from vis4d.common.util import init_random_seed
from vis4d.config import instantiate_classes
from vis4d.data.typing import DictData
from vis4d.engine.connectors import DataConnector
from vis4d.engine.loss_module import LossModule
from vis4d.engine.optim import Optimizer, set_up_optimizers
from vis4d.engine.util import ModelEMAAdapter


class TorchOptimizer(optim.Optimizer):
    """Wrapper around vis4d optimizer to make it compatible with pl."""

    def __init__(self, optimizer: Optimizer) -> None:
        """Creates a new Optimizer.

        Args:
            optimizer: The vis4d optimizer to wrap.
            model: The model to optimize.
        """
        self.optim = optimizer
        assert self.optim.optimizer is not None
        self._step = 0  # TODO: Check resume from checkpoint

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


class TrainingModule(pl.LightningModule):
    """LightningModule that wraps around the vis4d implementations.

    This is a wrapper around the vis4d implementations that allows to use
    pytorch-lightning for training and testing.
    """

    def __init__(
        self,
        model: ConfigDict,
        optimizers: list[ConfigDict],
        loss_module: None | LossModule,
        train_data_connector: None | DataConnector,
        test_data_connector: None | DataConnector,
        seed: None | int = None,
        use_ema_model_for_test: bool = False,
    ) -> None:
        """Initialize the TrainingModule.

        Args:
            model: The model config  to train.
            optimizers: The optimizers to use. Will be wrapped into a pytorch
                optimizer.
            loss_module: The loss function to use.
            train_data_connector: The data connector to use.
            test_data_connector: The data connector to use.
            data_connector: The data connector to use.
            seed (int, optional): The integer value seed for global random
                state. Defaults to None.
            use_ema_model_for_test (bool, optional): Whether to use the
                exponential moving average of the model for testing. Defaults
                to False.
        """
        super().__init__()
        self.model = model
        self.optims = optimizers
        self.loss_module = loss_module
        self.train_data_connector = train_data_connector
        self.test_data_connector = test_data_connector
        self.seed = seed
        self.use_ema_model_for_test = use_ema_model_for_test

    def setup(self, stage: str) -> None:
        """Setup the model."""
        if stage == "fit":
            if self.seed is None:
                seed = init_random_seed()
                seed = broadcast(seed)
            else:
                seed = self.seed

            seed_everything(seed, workers=True)
            rank_zero_info(f"Global seed set to {seed}")

        # Instantiate the model and optimizers after the seed has been set
        self.model = instantiate_classes(self.model)
        if stage == "fit":
            self.optims = set_up_optimizers(self.optims, self.model)

        # Set up the model EMA
        if self.use_ema_model_for_test:
            assert isinstance(
                self.model, ModelEMAAdapter
            ), "Model must be wrapped in ModelEMAAdapter"

    def forward(  # type: ignore # pylint: disable=arguments-differ
        self, data: DictData
    ) -> Any:
        """Forward pass through the model."""
        if self.training:
            assert self.train_data_connector is not None
            return self.model(**self.train_data_connector(data))
        assert self.test_data_connector is not None
        return self.model(**self.test_data_connector(data))

    def training_step(  # type: ignore # pylint: disable=arguments-differ,line-too-long,unused-argument
        self, batch: DictData, batch_idx: int
    ) -> Any:
        """Perform a single training step."""
        assert self.train_data_connector is not None
        out = self.model(**self.train_data_connector(batch))

        assert self.loss_module is not None
        losses = self.loss_module(out, batch)

        metrics = {}
        if isinstance(losses, Tensor):
            total_loss = losses
        else:
            total_loss = sum(list(losses.values()))
            for k, v in losses.items():
                metrics[k] = v.detach().cpu().item()

        metrics["loss"] = total_loss.detach().cpu().item()

        return {
            "loss": total_loss,
            "metrics": metrics,
            "predictions": out,
        }

    def validation_step(  # pylint: disable=arguments-differ,line-too-long,unused-argument
        self, batch: DictData, batch_idx: int, dataloader_idx: int = 0
    ) -> DictData:
        """Perform a single validation step."""
        assert self.test_data_connector is not None
        if self.use_ema_model_for_test:
            out = self.model.ema_model(**self.test_data_connector(batch))
        else:
            out = self.model(**self.test_data_connector(batch))
        return out

    def test_step(  # pylint: disable=arguments-differ,line-too-long,unused-argument
        self, batch: DictData, batch_idx: int, dataloader_idx: int = 0
    ) -> DictData:
        """Perform a single test step."""
        assert self.test_data_connector is not None
        if self.use_ema_model_for_test:
            out = self.model.ema_model(**self.test_data_connector(batch))
        else:
            out = self.model(**self.test_data_connector(batch))
        return out

    def configure_optimizers(self) -> list[TorchOptimizer]:
        """Return the optimizer to use."""
        return [TorchOptimizer(o) for o in self.optims]

    def optimizer_step(  # type: ignore # pylint: disable=arguments-differ
        self,
        epoch: int,
        batch_idx: int,
        optimizer: TorchOptimizer | LightningOptimizer,
        optimizer_closure: Callable[[], float] | None = None,
    ) -> None:
        """Perform a single optimization step."""
        optimizer.step(closure=optimizer_closure)

        # Update EMA model if available
        if isinstance(self.model, ModelEMAAdapter):
            self.model.update()
