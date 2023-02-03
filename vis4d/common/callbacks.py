"""This module contains utilities for callbacks."""
from __future__ import annotations

import logging
import os

from vis4d.common import ArgsType, MetricLogs
from vis4d.common.distributed import get_rank
from vis4d.common.typing import DictStrAny
from vis4d.data.typing import DictData
from vis4d.eval.base import Evaluator


class Callback:
    """Base class for Vis4D Callbacks."""

    def on_train_epoch_end(self) -> None:
        """Hook to run at the end of a training epoch."""

    def on_train_batch_end(self) -> None:
        """Hook to run at the end of a training batch."""

    def on_test_epoch_end(self) -> None:
        """Hook to run at the end of a testing epoch."""

    def on_test_batch_end(
        self, outputs: ArgsType, batch: ArgsType, eval_name: str
    ) -> None:
        """Hook to run at the end of a testing batch."""


def default_eval_connector(
    mode: str,  # pylint: disable=unused-argument
    data: DictData,
    outputs: ArgsType,
) -> DictStrAny:
    """Default eval connector forwards input and outputs."""
    return dict(data=data, outputs=outputs)


class EvaluatorCallback(Callback):
    """Callback for model evaluation."""

    def __init__(
        self,
        evaluator: Evaluator,
        eval_connector=default_eval_connector,
        output_dir: None | str = None,
        collect: str = "cpu",
    ) -> None:
        """Init callback."""
        assert collect in set(
            ("cpu", "gpu")
        ), f"Collect device {collect} unknown."
        self.logging_disabled = False
        self.collect = collect
        self.output_dir = output_dir
        self.evaluator = evaluator
        self.eval_connector = eval_connector
        self.logging_disabled = False
        self.run_eval = True

        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)

    def on_test_epoch_end(self) -> None:
        """Hook to run at the end of a testing epoch."""
        # TODO: need an all_gather function?
        # def gather_func(x):
        #     return all_gather_object_cpu(x)

        # self.evaluator.gather(gather_func)
        if get_rank() == 0:
            self.evaluate()
        self.evaluator.reset()

    def on_test_batch_end(
        self, outputs: ArgsType, batch: ArgsType, eval_name: str
    ) -> None:
        """Hook to run at the end of a testing batch."""
        eval_inputs = self.eval_connector.get_evaluator_input(
            eval_name, outputs, batch
        )
        self.evaluator.process(**eval_inputs)

    def evaluate(self) -> dict[str, MetricLogs]:
        """Evaluate the performance after processing all input/output pairs."""
        if not self.run_eval:
            return {}

        results = {}
        logger = logging.getLogger(__name__)
        if not self.logging_disabled:
            logger.info("Running evaluator %s...", str(self.evaluator))

        for metric in self.evaluator.metrics:
            if self.output_dir is not None:
                output_dir = os.path.join(self.output_dir, metric)
                os.makedirs(output_dir, exist_ok=True)
                self.evaluator.t(output_dir, metric)  # TODO implement save

            log_dict, log_str = self.evaluator.evaluate(metric)
            results[metric] = log_dict
            if not self.logging_disabled:
                for k, v in log_dict.items():
                    self.log(k, v, rank_zero_only=True)  # type: ignore # pylint: disable=no-member,line-too-long
                logger.info("Showing results for %s", metric)
                logger.info(log_str)
        return results
