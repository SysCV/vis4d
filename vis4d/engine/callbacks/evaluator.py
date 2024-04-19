"""This module contains utilities for callbacks."""

from __future__ import annotations

import os

from torch import nn

from vis4d.common import ArgsType, MetricLogs
from vis4d.common.distributed import (
    all_gather_object_cpu,
    broadcast,
    rank_zero_only,
    synchronize,
)
from vis4d.common.logging import rank_zero_info
from vis4d.data.typing import DictData
from vis4d.eval.base import Evaluator

from .base import Callback
from .trainer_state import TrainerState


class EvaluatorCallback(Callback):
    """Callback for model evaluation.

    Args:
        evaluator (Evaluator): Evaluator.
        metrics_to_eval (list[str], Optional): Metrics to evaluate. If None,
            all metrics in the evaluator will be evaluated. Defaults to None.
        save_predictions (bool): If the predictions should be saved. Defaults
            to False.
        save_prefix (str, Optional): Output directory for saving the
            evaluation results. Defaults to None.
    """

    def __init__(
        self,
        *args: ArgsType,
        evaluator: Evaluator,
        metrics_to_eval: list[str] | None = None,
        save_predictions: bool = False,
        save_prefix: None | str = None,
        **kwargs: ArgsType,
    ) -> None:
        """Init callback."""
        super().__init__(*args, **kwargs)
        self.evaluator = evaluator
        self.save_predictions = save_predictions
        self.metrics_to_eval = metrics_to_eval or self.evaluator.metrics

        if self.save_predictions:
            assert (
                save_prefix is not None
            ), "If save_predictions is True, save_prefix must be provided."
            self.output_dir = save_prefix

    def setup(self) -> None:  # pragma: no cover
        """Setup callback."""
        if self.save_predictions:
            self.output_dir = broadcast(self.output_dir)
            for metric in self.metrics_to_eval:
                output_dir = os.path.join(self.output_dir, metric)
                os.makedirs(output_dir, exist_ok=True)
        self.evaluator.reset()

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
        self.evaluator.process_batch(
            **self.get_test_callback_inputs(outputs, batch)
        )
        for metric in self.metrics_to_eval:
            # Save output predictions in current batch.
            if self.save_predictions:
                output_dir = os.path.join(self.output_dir, metric)
                self.evaluator.save_batch(metric, output_dir)

    def on_test_epoch_end(
        self, trainer_state: TrainerState, model: nn.Module
    ) -> None | MetricLogs:
        """Hook to run at the end of a testing epoch."""
        self.evaluator.gather(all_gather_object_cpu)

        synchronize()
        log_dict = self.evaluate()
        log_dict = broadcast(log_dict)
        self.evaluator.reset()
        return log_dict

    @rank_zero_only
    def evaluate(self) -> MetricLogs:
        """Evaluate the performance after processing all input/output pairs.

        Returns:
            MetricLogs: A dictionary containing the evaluation results. The
                keys are formatted as {metric_name}/{key_name}, and the
                values are the corresponding evaluated values.
        """
        rank_zero_info("Running evaluator %s...", str(self.evaluator))
        self.evaluator.process()

        log_dict = {}
        for metric in self.metrics_to_eval:
            # Save output predictions. This is done here instead of
            # on_test_batch_end because the evaluator may not have processed
            # all batches yet.
            if self.save_predictions:
                output_dir = os.path.join(self.output_dir, metric)
                self.evaluator.save(metric, output_dir)

            # Evaluate metric
            metric_dict, metric_str = self.evaluator.evaluate(metric)
            for k, v in metric_dict.items():
                log_k = metric + "/" + k
                rank_zero_info("%s: %.4f", log_k, v)
                log_dict[f"{metric}/{k}"] = v

            rank_zero_info("Showing results for metric: %s", metric)
            rank_zero_info(metric_str)

        return log_dict
