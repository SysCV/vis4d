"""Wrapper to connect vis4d callbacks to pytorch lightning callbacks."""


from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
from torch import nn

from vis4d.common.callbacks import Callback
from vis4d.engine.connectors import DataConnector
from vis4d.pl.training_module import TrainingModule


def get_model(model: pl.LightningModule) -> nn.Module:
    """Get model from pl module."""
    if isinstance(model, TrainingModule):
        return model.model
    return model


class CallbackWrapper(pl.Callback):
    """Wrapper to connect vis4d callbacks to pytorch lightning callbacks."""

    def __init__(
        self,
        callback: Callback,
        data_connector: DataConnector,
        callback_key: str,
    ) -> None:
        """Init class."""
        self.callback = callback
        self.data_connector = data_connector
        self.callback_key = callback_key

    def on_test_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Wait for on_test_epoch_end PL hook to call 'evaluate'."""
        if self.callback.run_on_epoch(pl_module.current_epoch):
            self.callback.on_test_epoch_end(
                get_model(pl_module), pl_module.current_epoch
            )

    def on_test_batch_end(  # type: ignore
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Wait for on_test_batch_end PL hook to call 'process'."""
        if self.callback.run_on_epoch(pl_module.current_epoch):
            self.callback.on_test_batch_end(
                model=get_model(pl_module),
                inputs=self.data_connector.get_callback_input(
                    self.callback_key, outputs, batch, cb_type="test"
                ),
            )

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Wait for on_validation_epoch_end PL hook to call 'evaluate'."""
        if self.callback.run_on_epoch(pl_module.current_epoch):
            self.callback.on_test_epoch_end(
                get_model(pl_module), pl_module.current_epoch
            )

    def on_validation_batch_end(  # type: ignore
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Wait for on_validation_batch_end PL hook to call 'process'."""
        if self.callback.run_on_epoch(pl_module.current_epoch):
            self.callback.on_test_batch_end(
                model=get_model(pl_module),
                inputs=self.data_connector.get_callback_input(
                    self.callback_key, outputs, batch, cb_type="test"
                ),
            )

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Hook to run at the end of a training epoch."""
        if self.callback.run_on_epoch(pl_module.current_epoch):
            self.callback.on_train_epoch_end(
                get_model(pl_module), pl_module.current_epoch
            )

    def on_train_batch_end(  # type: ignore
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Hook to run at the end of a training batch."""
        if self.callback.run_on_epoch(pl_module.current_epoch):
            model_pred = outputs["predictions"]
            self.callback.on_train_batch_end(
                model=get_model(pl_module),
                inputs=self.data_connector.get_callback_input(
                    self.callback_key, model_pred, batch, cb_type="train"
                ),
            )
