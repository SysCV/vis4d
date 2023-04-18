"""This module contains utilities for callbacks."""
from __future__ import annotations

from collections import defaultdict

import torch
from torch import nn

from vis4d.common import DictStrAny
from vis4d.common.logging import rank_zero_info
from vis4d.common.progress import compose_log_str
from vis4d.common.time import Timer

from .base import Callback


class LoggingCallback(Callback):
    """Callback for logging."""

    def __init__(self, refresh_rate: int = 50) -> None:
        """Init callback."""
        super().__init__(1, -1)
        self._refresh_rate = refresh_rate
        self._metrics: dict[str, list[torch.Tensor]] = defaultdict(list)
        self.timer = Timer()

    def on_train_epoch_start(self, model: nn.Module, epoch: int) -> None:
        """Hook to run at the start of a training epoch."""
        self.timer.reset()

    def on_train_batch_end(
        self,
        model: nn.Module,
        shared_inputs: DictStrAny,
        inputs: DictStrAny,
    ) -> None:
        """Hook to run at the end of a training batch."""
        for k, v in shared_inputs["metrics"].items():
            self._metrics[k].append(v)
        if shared_inputs["cur_iter"] % self._refresh_rate == 0:
            rank_zero_info(
                compose_log_str(
                    f"Epoch {shared_inputs['epoch'] + 1}",
                    shared_inputs["cur_iter"] + 1,
                    shared_inputs["total_iters"],
                    self.timer,
                    {
                        k: sum(v) / len(v) if len(v) > 0 else float("NaN")
                        for k, v in self._metrics.items()
                    },
                )
            )
            self._metrics = defaultdict(list)

    def on_test_epoch_start(self, model: nn.Module, epoch: int) -> None:
        """Hook to run at the start of a training epoch."""
        self.timer.reset()

    def on_test_batch_end(
        self,
        model: nn.Module,
        shared_inputs: DictStrAny,
        inputs: DictStrAny,
    ) -> None:
        """Hook to run at the end of a training batch."""
        if shared_inputs["cur_iter"] % self._refresh_rate == 0:
            rank_zero_info(
                compose_log_str(
                    "Testing",
                    shared_inputs["cur_iter"] + 1,
                    shared_inputs["total_iters"],
                    self.timer,
                )
            )
