"""This module contains utilities for callbacks."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import lightning.pytorch as pl

from vis4d.common import ArgsType, MetricLogs
from vis4d.common.logging import rank_zero_info
from vis4d.common.progress import compose_log_str
from vis4d.common.time import Timer

from .base import Callback


class LoggingCallback(Callback):
    """Callback for logging."""

    def __init__(
        self, *args: ArgsType, refresh_rate: int = 50, **kwargs: ArgsType
    ) -> None:
        """Init callback."""
        super().__init__(*args, **kwargs)
        self._refresh_rate = refresh_rate
        self._metrics: dict[str, list[float]] = defaultdict(list)
        self.train_timer = Timer()
        self.test_timer = Timer()
        self.last_step = 0

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Hook to run at the start of a training epoch."""
        if self.epoch_based:
            self.train_timer.reset()
            self.last_step = 0
            self._metrics.clear()
        elif trainer.global_step == 0:
            self.train_timer.reset()

    def on_train_batch_start(  # type: ignore
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Hook to run at the start of a training batch."""
        if self.train_timer.paused:
            self.train_timer.resume()

    def on_train_batch_end(  # type: ignore
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Hook to run at the end of a training batch."""
        if "metrics" in outputs:
            for k, v in outputs["metrics"].items():
                self._metrics[k].append(v)

        if self.epoch_based:
            cur_iter = batch_idx + 1

            # Resolve float("inf") to -1
            if isinstance(trainer.num_training_batches, float):
                total_iters = -1
            else:
                total_iters = trainer.num_training_batches
        else:
            cur_iter = trainer.global_step + 1
            total_iters = trainer.max_steps

        if cur_iter % self._refresh_rate == 0 and cur_iter != self.last_step:
            prefix = (
                f"Epoch {pl_module.current_epoch + 1}"
                if self.epoch_based
                else "Iter"
            )

            log_dict: MetricLogs = {
                k: sum(v) / len(v) if len(v) > 0 else float("NaN")
                for k, v in self._metrics.items()
            }

            rank_zero_info(
                compose_log_str(
                    prefix, cur_iter, total_iters, self.train_timer, log_dict
                )
            )

            self._metrics.clear()
            self.last_step = cur_iter

            for k, v in log_dict.items():
                pl_module.log(f"train/{k}", v, rank_zero_only=True)

    def on_validation_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Hook to run at the start of a validation epoch."""
        self.test_timer.reset()
        self.train_timer.pause()

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
        cur_iter = batch_idx + 1

        # Resolve float("inf") to -1
        if isinstance(trainer.num_val_batches[dataloader_idx], int):
            total_iters = int(trainer.num_val_batches[dataloader_idx])
        else:
            total_iters = -1

        if cur_iter % self._refresh_rate == 0:
            rank_zero_info(
                compose_log_str(
                    "Validation", cur_iter, total_iters, self.test_timer
                )
            )

    def on_test_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Hook to run at the start of a testing epoch."""
        self.test_timer.reset()
        self.train_timer.pause()

    def on_test_batch_end(  # type: ignore
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Hook to run at the end of a testing batch."""
        cur_iter = batch_idx + 1

        # Resolve float("inf") to -1
        if isinstance(trainer.num_test_batches[dataloader_idx], int):
            total_iters = int(trainer.num_test_batches[dataloader_idx])
        else:
            total_iters = -1

        if cur_iter % self._refresh_rate == 0:
            rank_zero_info(
                compose_log_str(
                    "Testing", cur_iter, total_iters, self.test_timer
                )
            )
