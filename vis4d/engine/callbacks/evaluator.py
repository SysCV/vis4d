"""This module contains utilities for callbacks."""
from __future__ import annotations

import os

from torch import nn

from vis4d.common import ArgsType, MetricLogs
from vis4d.common.distributed import all_gather_object_cpu, broadcast, get_rank
from vis4d.common.logging import rank_zero_info
from vis4d.data.typing import DictData
from vis4d.eval.base import Evaluator

from .base import Callback
from .trainer_state import TrainerState


class EvaluatorCallback(Callback):
    """Callback for model evaluation."""

    def __init__(
        self,
        *args: ArgsType,
        evaluator: Evaluator,
        metrics: None | list[str] = None,
        save_predictions: bool = False,
        save_prefix: None | str = None,
        **kwargs: ArgsType,
    ) -> None:
        """Init callback.

        Args:
            evaluator (Evaluator): Evaluator to use.
            metrics (list[str], Optional): Metrics to evaluate. Defaults to
                None, which means all available metrics will be evaluated.
            save_predictions (bool): If the predictions should be saved.
                Defaults to False.
            save_prefix (str, Optional): Output directory for saving the
                evaluation results. Defaults to None.
        """
        super().__init__(*args, **kwargs)
        self.evaluator = evaluator
        self.metrics = metrics
        self.save_predictions = save_predictions

        if self.save_predictions:
            assert (
                save_prefix is not None
            ), "If save_predictions is True, save_prefix must be provided."
            self.output_dir = save_prefix

    def setup(self) -> None:  # pragma: no cover
        """Setup callback."""
        if self.save_predictions:
            self.output_dir = broadcast(self.output_dir)
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
            **self.get_data_connector_results(outputs, batch, train=False)
        )
        if self.save_predictions:
            metrics = (
                self.evaluator.metrics
                if self.metrics is None
                else self.metrics
            )
            for metric in metrics:
                self.evaluator.save_batch(metric, self.output_dir)

    def on_test_epoch_end(
        self, trainer_state: TrainerState, model: nn.Module
    ) -> None | MetricLogs:
        """Hook to run at the end of a testing epoch."""
        self.evaluator.gather(all_gather_object_cpu)
        self.evaluator.process()
        if get_rank() == 0:
            log_dict = self.evaluate()
        else:  # pragma: no cover
            log_dict = None
        log_dict = broadcast(log_dict)
        self.evaluator.reset()
        return log_dict

    def evaluate(self) -> MetricLogs:
        """Evaluate the performance after processing all input/output pairs."""
        rank_zero_info("Running evaluator %s...", str(self.evaluator))

        self.evaluator.process()

        for metric in self.evaluator.metrics:
            # Save output
            if self.save_predictions:
                output_dir = os.path.join(self.output_dir, metric)
                os.makedirs(output_dir, exist_ok=True)

            # Evaluate metric
            log_dict, log_str = self.evaluator.evaluate(metric)
            self.evaluator.save(metric, output_dir)
            for k, v in log_dict.items():
                rank_zero_info("%s: %.4f", k, v)
            rank_zero_info("Showing evaluation results for %s", metric)
            rank_zero_info(log_str)
        return log_dict