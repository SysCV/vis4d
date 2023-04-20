"""This module contains utilities for callbacks."""
from __future__ import annotations

from collections import defaultdict

import torch
from torch import nn

from vis4d.common import ArgsType
from vis4d.common.logging import rank_zero_info
from vis4d.common.progress import compose_log_str
from vis4d.common.time import Timer

from vis4d.data.typing import DictData

from .base import Callback, CallbackInputs


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
        self._metrics: dict[str, list[torch.Tensor]] = defaultdict(list)
        self.timer = Timer()

    def on_train_epoch_start(
        self, callback_inputs: CallbackInputs, model: nn.Module
    ) -> None:
        """Hook to run at the start of a training epoch."""
        self.timer.reset()

    def on_train_batch_end(
        self,
        callback_inputs: CallbackInputs,
        model: nn.Module,
        predictions: DictData,
        data: DictData,
    ) -> None:
        """Hook to run at the end of a training batch."""
        if "metrics" in callback_inputs:
            for k, v in callback_inputs["metrics"].items():
                self._metrics[k].append(v)
        if callback_inputs["cur_iter"] % self._refresh_rate == 0:
            rank_zero_info(
                compose_log_str(
                    f"Epoch {callback_inputs['epoch'] + 1}",
                    callback_inputs["cur_iter"] + 1,
                    callback_inputs["total_iters"],
                    self.timer,
                    {
                        k: sum(v) / len(v) if len(v) > 0 else float("NaN")
                        for k, v in self._metrics.items()
                    },
                )
            )
            self._metrics = defaultdict(list)

    def on_test_epoch_start(
        self, callback_inputs: CallbackInputs, model: nn.Module
    ) -> None:
        """Hook to run at the start of a training epoch."""
        self.timer.reset()

    def on_test_batch_end(
        self,
        callback_inputs: CallbackInputs,
        model: nn.Module,
        predictions: DictData,
        data: DictData,
    ) -> None:
        """Hook to run at the end of a training batch."""
        if callback_inputs["cur_iter"] % self._refresh_rate == 0:
            rank_zero_info(
                compose_log_str(
                    "Testing",
                    callback_inputs["cur_iter"] + 1,
                    callback_inputs["total_iters"],
                    self.timer,
                )
            )
