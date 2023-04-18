"""This module contains utilities for callbacks."""
from __future__ import annotations

import os

from torch import nn

from vis4d.common import DictStrAny, MetricLogs
from vis4d.common.distributed import all_gather_object_cpu, broadcast, get_rank
from vis4d.common.logging import rank_zero_info
from vis4d.eval.base import Evaluator

from .base import Callback


class EvaluatorCallback(Callback):
    """Callback for model evaluation."""

    def __init__(
        self,
        evaluator: Evaluator,
        save_prefix: None | str = None,
        collect: str = "cpu",
        run_every_nth_epoch: int = 1,
        num_epochs: int = -1,
    ) -> None:
        """Init callback.

        Args:
            evaluator (Evaluator): Evaluator.
            save_prefix (str, Optional): Output directory for saving the
                evaluation results. Defaults to None (no save).
            collect (str): Which device to collect results across GPUs on.
                Defaults to "cpu".
            run_every_nth_epoch (int): Evaluate model every nth epoch.
                Defaults to 1.
            num_epochs (int): Number of total epochs, used for determining
                whether to evaluate at the final epoch. Defaults to -1.
        """
        super().__init__(run_every_nth_epoch, num_epochs)
        assert collect in set(
            ("cpu", "gpu")
        ), f"Collect device {collect} unknown."
        self.collect = collect
        self.output_dir = save_prefix
        self.evaluator = evaluator

    def setup(self) -> None:  # pragma: no cover
        """Setup callback."""
        self.output_dir = broadcast(self.output_dir)
        self.evaluator.reset()

    def on_test_epoch_end(
        self, model: nn.Module, epoch: None | int = None
    ) -> None | MetricLogs:
        """Hook to run at the end of a testing epoch."""
        self.evaluator.gather(all_gather_object_cpu)
        if get_rank() == 0:
            log_dict = self.evaluate()
        else:  # pragma: no cover
            log_dict = None
        log_dict = broadcast(log_dict)
        self.evaluator.reset()
        return log_dict

    def on_test_batch_end(
        self, model: nn.Module, shared_inputs: DictStrAny, inputs: DictStrAny
    ) -> None:
        """Hook to run at the end of a testing batch."""
        self.evaluator.process(**inputs)

    def evaluate(self) -> MetricLogs:
        """Evaluate the performance after processing all input/output pairs."""
        rank_zero_info("Running evaluator %s...", str(self.evaluator))

        for metric in self.evaluator.metrics:
            # Save output
            if self.output_dir is not None:
                output_dir = os.path.join(self.output_dir, metric)
                os.makedirs(output_dir, exist_ok=True)
                self.evaluator.save(metric, output_dir)

            # Evaluate metric
            log_dict, log_str = self.evaluator.evaluate(metric)
            for k, v in log_dict.items():
                rank_zero_info("%s: %.3f", k, v)
            rank_zero_info("Showing results for %s", metric)
            rank_zero_info(log_str)
        return log_dict
