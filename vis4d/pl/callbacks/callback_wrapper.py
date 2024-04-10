"""Wrapper to connect PyTorch Lightning callbacks."""

from __future__ import annotations

from typing import Any

import lightning.pytorch as pl
from torch import nn

from vis4d.engine.callbacks import Callback, TrainerState
from vis4d.engine.loss_module import LossModule
from vis4d.pl.training_module import TrainingModule


def get_trainer_state(
    trainer: pl.Trainer, pl_module: pl.LightningModule, val: bool = False
) -> TrainerState:
    """Wrap pl.Trainer and pl.LightningModule into Trainer."""
    # Resolve float("inf") to -1
    if val:
        test_dataloader = trainer.val_dataloaders
        num_test_batches = [
            num_batch if isinstance(num_batch, int) else -1
            for num_batch in trainer.num_val_batches
        ]
    else:
        test_dataloader = trainer.test_dataloaders
        num_test_batches = [
            num_batch if isinstance(num_batch, int) else -1
            for num_batch in trainer.num_test_batches
        ]

    # Map max_epochs=None to -1
    if trainer.max_epochs is None:
        num_epochs = -1
    else:
        num_epochs = trainer.max_epochs

    # Resolve float("inf") to -1
    if isinstance(trainer.num_training_batches, float):
        num_train_batches = -1
    else:
        num_train_batches = trainer.num_training_batches

    return TrainerState(
        current_epoch=pl_module.current_epoch,
        num_epochs=num_epochs,
        global_step=trainer.global_step,
        num_steps=trainer.max_steps,
        train_dataloader=trainer.train_dataloader,
        num_train_batches=num_train_batches,
        test_dataloader=test_dataloader,
        num_test_batches=num_test_batches,
        train_module=trainer,
        train_engine="pl",
    )


def get_model(model: pl.LightningModule) -> nn.Module:
    """Get model from pl module."""
    if isinstance(model, TrainingModule):
        return model.model
    return model


def get_loss_module(loss_module: pl.LightningModule) -> LossModule:
    """Get loss_module from pl module."""
    if isinstance(loss_module, TrainingModule):
        assert loss_module.loss_module is not None
        return loss_module.loss_module
    return loss_module  # type: ignore


class CallbackWrapper(pl.Callback):
    """Wrapper to connect vis4d callbacks to pytorch lightning callbacks."""

    def __init__(self, callback: Callback) -> None:
        """Init class."""
        self.callback = callback

    def setup(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str
    ) -> None:
        """Setup callback."""
        self.callback.setup()

    def on_train_batch_start(  # type: ignore
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Called when the train batch begins."""
        trainer_state = get_trainer_state(trainer, pl_module)

        self.callback.on_train_batch_start(
            trainer_state=trainer_state,
            model=get_model(pl_module),
            loss_module=get_loss_module(pl_module),
            batch=batch,
            batch_idx=batch_idx,
        )

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Hook to run at the start of a training epoch."""
        self.callback.on_train_epoch_start(
            get_trainer_state(trainer, pl_module),
            get_model(pl_module),
            get_loss_module(pl_module),
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
        trainer_state = get_trainer_state(trainer, pl_module)
        trainer_state["metrics"] = outputs["metrics"]

        log_dict = self.callback.on_train_batch_end(
            trainer_state=trainer_state,
            model=get_model(pl_module),
            loss_module=get_loss_module(pl_module),
            outputs=outputs["predictions"],
            batch=batch,
            batch_idx=batch_idx,
        )

        if log_dict is not None:
            for k, v in log_dict.items():
                pl_module.log(f"train/{k}", v, rank_zero_only=True)

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Hook to run at the end of a training epoch."""
        self.callback.on_train_epoch_end(
            get_trainer_state(trainer, pl_module),
            get_model(pl_module),
            get_loss_module(pl_module),
        )

    def on_validation_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Hook to run at the start of a validation epoch."""
        self.callback.on_test_epoch_start(
            get_trainer_state(trainer, pl_module, val=True),
            get_model(pl_module),
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
        self.callback.on_test_batch_end(
            trainer_state=get_trainer_state(trainer, pl_module, val=True),
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
        log_dict = self.callback.on_test_epoch_end(
            get_trainer_state(trainer, pl_module, val=True),
            get_model(pl_module),
        )

        if log_dict is not None:
            for k, v in log_dict.items():
                pl_module.log(
                    f"val/{k}", v, sync_dist=True, rank_zero_only=True
                )

    def on_test_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Hook to run at the start of a testing epoch."""
        self.callback.on_test_epoch_start(
            get_trainer_state(trainer, pl_module), get_model(pl_module)
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
        self.callback.on_test_batch_end(
            trainer_state=get_trainer_state(trainer, pl_module),
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
        log_dict = self.callback.on_test_epoch_end(
            get_trainer_state(trainer, pl_module), get_model(pl_module)
        )

        if log_dict is not None:
            for k, v in log_dict.items():
                pl_module.log(
                    f"test/{k}", v, sync_dist=True, rank_zero_only=True
                )
