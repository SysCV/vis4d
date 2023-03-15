"""Wrapper to connect vis4d callbacks to pytorch lightning callbacks."""
from __future__ import annotations

from collections.abc import Iterable

import pytorch_lightning as pl


class OptimEpochCallback(pl.Callback):
    """Callback to step optimizer at the end of each epoch."""

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Hook to run at the end of a training epoch."""
        optimizers = pl_module.optimizers()
        if not isinstance(optimizers, Iterable):
            optimizers = [optimizers]

        for optimizer in optimizers:
            optimizer.step_on_epoch(pl_module.current_epoch)  # type: ignore
