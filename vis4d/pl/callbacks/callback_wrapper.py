"""Wrapper to connect PyTorch Lightning callbacks."""
from __future__ import annotations

from typing import Any

import lightning.pytorch as pl
from torch import nn

from vis4d.common.callbacks import Callback
from vis4d.engine.trainer import Trainer
from vis4d.pl.training_module import TrainingModule


def get_trainer(trainer: pl.Trainer, pl_module: pl.LightningModule) -> Trainer:
    """Wrap pl.Trainer and pl.LightningModule into Trainer."""
    trainer = Trainer(
        num_epochs=trainer.max_epochs,
        data_connector=pl_module.data_connector,
        train_dataloader=trainer.train_dataloader,
        num_train_batches=trainer.num_training_batches,
        test_dataloader=trainer.test_dataloaders,
        num_test_batches=trainer.num_test_batches,
        epoch=pl_module.current_epoch,
        global_steps=trainer.global_step,
    )
    return trainer


def get_model(model: pl.LightningModule) -> nn.Module:
    """Get model from pl module."""
    if isinstance(model, TrainingModule):
        return model.model
    return model


class CallbackWrapper(pl.Callback):  # type: ignore
    """Wrapper to connect vis4d callbacks to pytorch lightning callbacks."""

    def __init__(self, callback: Callback) -> None:
        """Init class."""
        self.callback = callback

    def setup(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str
    ) -> None:
        """Setup callback."""
        self.callback.setup()

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Hook to run at the start of a training epoch."""
        if self.callback.run_on_epoch(pl_module.current_epoch):
            self.callback.on_train_epoch_start(
                get_trainer(trainer, pl_module), get_model(pl_module)
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
            trainer_ = get_trainer(trainer, pl_module)
            trainer_.metrics = outputs["metrics"]

            self.callback.on_train_batch_end(
                trainer=trainer_,
                model=get_model(pl_module),
                outputs=outputs["predictions"],
                batch=batch,
                batch_idx=batch_idx,
            )

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Hook to run at the end of a training epoch."""
        if self.callback.run_on_epoch(pl_module.current_epoch):
            self.callback.on_train_epoch_end(
                trainer=get_trainer(trainer, pl_module),
                model=get_model(pl_module),
            )

    def on_validation_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Hook to run at the start of a validation epoch."""
        if self.callback.run_on_epoch(pl_module.current_epoch):
            self.callback.on_test_epoch_start(
                get_trainer(trainer, pl_module), get_model(pl_module)
            )

    def on_validation_batch_end(  # type: ignore
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Wait for on_validation_batch_end PL hook to call 'process'."""
        if self.callback.run_on_epoch(pl_module.current_epoch):
            self.callback.on_test_batch_end(
                trainer=get_trainer(trainer, pl_module),
                model=get_model(pl_module),
                outputs=outputs,
                batch=batch,
                batch_idx=batch_idx,
                dataloader_idx=dataloader_idx,
            )

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Wait for on_validation_epoch_end PL hook to call 'evaluate'."""
        if self.callback.run_on_epoch(pl_module.current_epoch):
            log_dict = self.callback.on_test_epoch_end(
                get_trainer(trainer, pl_module), get_model(pl_module)
            )

            if log_dict is not None:
                for k, v in log_dict.items():
                    pl_module.log(k, v, rank_zero_only=True, sync_dist=True)

    def on_test_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Hook to run at the start of a testing epoch."""
        if self.callback.run_on_epoch(pl_module.current_epoch):
            self.callback.on_test_epoch_start(
                get_trainer(trainer, pl_module), get_model(pl_module)
            )

    def on_test_batch_end(  # type: ignore
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Wait for on_test_batch_end PL hook to call 'process'."""
        if self.callback.run_on_epoch(pl_module.current_epoch):
            self.callback.on_test_batch_end(
                trainer=get_trainer(trainer, pl_module),
                model=get_model(pl_module),
                outputs=outputs,
                batch=batch,
                batch_idx=batch_idx,
                dataloader_idx=dataloader_idx,
            )

    def on_test_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Wait for on_test_epoch_end PL hook to call 'evaluate'."""
        if self.callback.run_on_epoch(pl_module.current_epoch):
            log_dict = self.callback.on_test_epoch_end(
                get_trainer(trainer, pl_module), get_model(pl_module)
            )

            if log_dict is not None:
                for k, v in log_dict.items():
                    pl_module.log(k, v, rank_zero_only=True, sync_dist=True)
