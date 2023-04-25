"""This module contains utilities for callbacks."""
from __future__ import annotations

from collections import defaultdict

from torch import nn, Tensor

from vis4d.common import ArgsType, TrainerType
from vis4d.common.logging import rank_zero_info
from vis4d.common.progress import compose_log_str
from vis4d.common.time import Timer

from vis4d.data.typing import DictData

from .base import Callback


class LoggingCallback(Callback):
    """Callback for logging."""

    def __init__(
        self,
        *args: ArgsType,
        refresh_rate: int = 50,
        **kwargs: ArgsType,
    ) -> None:
        """Init callback."""
        super().__init__(*args, **kwargs)
        self._refresh_rate = refresh_rate
        self._metrics: dict[str, list[Tensor]] = defaultdict(list)
        self.timer = Timer()

    def on_train_epoch_start(
        self, trainer: TrainerType, model: nn.Module
    ) -> None:
        """Hook to run at the start of a training epoch."""
        self.timer.reset()

    def on_train_batch_end(
        self,
        trainer: TrainerType,
        model: nn.Module,
        outputs: DictData,
        batch: DictData,
        batch_idx: int,
    ) -> None:
        """Hook to run at the end of a training batch."""
        # TODO: Check this mismatch with PL logger
        for k, v in trainer.metrics.items():
            self._metrics[k].append(v)
        if batch_idx % self._refresh_rate == 0:
            rank_zero_info(
                compose_log_str(
                    f"Epoch {trainer.epoch + 1}",
                    batch_idx + 1,
                    trainer.num_train_batches,
                    self.timer,
                    {
                        k: sum(v) / len(v) if len(v) > 0 else float("NaN")
                        for k, v in self._metrics.items()
                    },
                )
            )
            self._metrics = defaultdict(list)

    def on_test_epoch_start(
        self, trainer: TrainerType, model: nn.Module
    ) -> None:
        """Hook to run at the start of a training epoch."""
        self.timer.reset()

    def on_test_batch_end(
        self,
        trainer: TrainerType,
        model: nn.Module,
        outputs: DictData,
        batch: DictData,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Hook to run at the end of a training batch."""
        if batch_idx % self._refresh_rate == 0:
            rank_zero_info(
                compose_log_str(
                    "Testing",
                    batch_idx + 1,
                    trainer.num_test_batches[dataloader_idx],
                    self.timer,
                )
            )
