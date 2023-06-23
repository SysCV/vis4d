"""Wrapper to connect vis4d callbacks to pytorch lightning callbacks."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import lightning.pytorch as pl

from vis4d.engine.optim.optimizer import warmup_step
from vis4d.pl.training_module import TrainingModule


class LRWarmUpCallback(pl.Callback):
    """Callback to set up learning warmup during training."""

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Hook to run at the end of a training epoch."""
        optimizers = pl_module.optimizers()
        if not isinstance(optimizers, Iterable):
            optimizers = [optimizers]

        for i, _ in enumerate(optimizers):
            assert isinstance(pl_module, TrainingModule)
            lr_warmup = pl_module.lr_warmups[i]
            if lr_warmup is not None and lr_warmup["epoch_based"]:
                warmup_step(
                    pl_module.current_epoch,
                    lr_warmup["warmup"],
                    optimizers[i],  # type: ignore
                )

    def on_train_batch_start(  # type: ignore
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Called when fit begins."""
        optimizers = pl_module.optimizers()
        if not isinstance(optimizers, Iterable):
            optimizers = [optimizers]

        for i, _ in enumerate(optimizers):
            assert isinstance(pl_module, TrainingModule)
            lr_warmup = pl_module.lr_warmups[i]
            if lr_warmup is not None and not lr_warmup["epoch_based"]:
                warmup_step(
                    trainer.global_step,
                    lr_warmup["warmup"],
                    optimizers[i],  # type: ignore
                )
