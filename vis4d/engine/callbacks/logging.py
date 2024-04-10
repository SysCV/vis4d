"""This module contains utilities for callbacks."""

from __future__ import annotations

from collections import defaultdict

from torch import nn

from vis4d.common import ArgsType, MetricLogs
from vis4d.common.logging import rank_zero_info
from vis4d.common.progress import compose_log_str
from vis4d.common.time import Timer
from vis4d.data.typing import DictData
from vis4d.engine.loss_module import LossModule

from .base import Callback
from .trainer_state import TrainerState


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

    def on_train_batch_start(
        self,
        trainer_state: TrainerState,
        model: nn.Module,
        loss_module: LossModule,
        batch: DictData,
        batch_idx: int,
    ) -> None:
        """Hook to run at the start of a training batch."""
        if not self.epoch_based and self.train_timer.paused:
            self.train_timer.resume()

    def on_train_epoch_start(
        self,
        trainer_state: TrainerState,
        model: nn.Module,
        loss_module: LossModule,
    ) -> None:
        """Hook to run at the start of a training epoch."""
        if self.epoch_based:
            self.train_timer.reset()
            self.last_step = 0
            self._metrics.clear()
        elif trainer_state["global_step"] == 0:
            self.train_timer.reset()

    def on_train_batch_end(
        self,
        trainer_state: TrainerState,
        model: nn.Module,
        loss_module: LossModule,
        outputs: DictData,
        batch: DictData,
        batch_idx: int,
    ) -> None | MetricLogs:
        """Hook to run at the end of a training batch."""
        if "metrics" in trainer_state:
            for k, v in trainer_state["metrics"].items():
                self._metrics[k].append(v)

        if self.epoch_based:
            cur_iter = batch_idx + 1

            total_iters = (
                trainer_state["num_train_batches"]
                if trainer_state["num_train_batches"] is not None
                else -1
            )
        else:
            # After optimizer.step(), global_step is already incremented by 1.
            cur_iter = trainer_state["global_step"]
            total_iters = trainer_state["num_steps"]

        log_dict: None | MetricLogs = None
        if cur_iter % self._refresh_rate == 0 and cur_iter != self.last_step:
            prefix = (
                f"Epoch {trainer_state['current_epoch'] + 1}"
                if self.epoch_based
                else "Iter"
            )

            log_dict = {
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

        return log_dict

    def on_test_epoch_start(
        self, trainer_state: TrainerState, model: nn.Module
    ) -> None:
        """Hook to run at the start of a testing epoch."""
        self.test_timer.reset()
        if not self.epoch_based:
            self.train_timer.pause()

    def on_test_batch_end(
        self,
        trainer_state: TrainerState,
        model: nn.Module,
        outputs: DictData,
        batch: DictData,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Hook to run at the end of a testing batch."""
        cur_iter = batch_idx + 1
        total_iters = (
            trainer_state["num_test_batches"][dataloader_idx]
            if trainer_state["num_test_batches"] is not None
            else -1
        )

        if cur_iter % self._refresh_rate == 0:
            rank_zero_info(
                compose_log_str(
                    "Testing", cur_iter, total_iters, self.test_timer
                )
            )
