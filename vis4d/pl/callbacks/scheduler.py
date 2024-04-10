"""Callback to configure learning rate during training."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import lightning.pytorch as pl

from vis4d.engine.optim.scheduler import LRSchedulerWrapper


class LRSchedulerCallback(pl.Callback):
    """Callback to configure learning rate during training."""

    def on_train_batch_end(  # type: ignore
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Hook on training batch end."""
        schedulers = pl_module.lr_schedulers()

        if not isinstance(schedulers, Iterable):
            schedulers = [schedulers]  # type: ignore

        for scheduler in schedulers:
            if scheduler is None:
                continue
            assert isinstance(scheduler, LRSchedulerWrapper)
            scheduler.step_on_batch(trainer.global_step)
