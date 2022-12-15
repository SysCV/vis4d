# pylint: disable=consider-alternative-union-syntax,consider-using-alias
"""Evaluation components for tracking."""
import logging
import os
from typing import Any, Dict, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT

from vis4d.common import MetricLogs
from vis4d.common.typing import DictStrAny
from vis4d.data.datasets.base import DictData
from vis4d.eval.base import Evaluator
from vis4d.pl.distributed import all_gather_object_cpu


def default_eval_connector(
    mode: str, data: DictData, outputs  # pylint: disable=unused-argument
) -> DictStrAny:
    """Default eva connector forwards input and outputs."""
    return dict(data=data, outputs=outputs)


class DefaultEvaluatorCallback(Callback):
    """Base class for Vis4D Evaluators.

    This class will accumulate the inputs/outputs in 'process', and produce
    evaluation results in 'evaluate'.
    """

    def __init__(
        self,
        dataloader_idx: int,
        evaluator: Evaluator,
        eval_connector=default_eval_connector,
        output_dir: Optional[str] = None,
        collect: str = "cpu",
    ) -> None:
        """Init class."""
        assert collect in set(
            ("cpu", "gpu")
        ), f"Collect device {collect} unknown."
        self.logging_disabled = False
        self.collect = collect
        self.dataloader_idx = dataloader_idx
        self.output_dir = output_dir
        self.evaluator = evaluator
        self.eval_connector = eval_connector
        self.logging_disabled = False
        self.run_eval = True

        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)

    def on_test_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Wait for on_test_epoch_end PL hook to call 'evaluate'."""

        def gather_func(x):
            return all_gather_object_cpu(x, pl_module)

        self.evaluator.gather(gather_func)
        if trainer.is_global_zero:
            self.evaluate()
        self.evaluator.reset()

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Wait for on_validation_epoch_end PL hook to call 'evaluate'."""

        def gather_func(x):
            return all_gather_object_cpu(x, pl_module)

        self.evaluator.gather(gather_func)
        if trainer.is_global_zero:
            self.evaluate()
        self.evaluator.reset()

    def on_test_batch_end(  # type: ignore
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Wait for on_test_batch_end PL hook to call 'process'."""
        if dataloader_idx == self.dataloader_idx:
            eval_inputs = self.eval_connector("", batch, outputs)
            self.evaluator.process(**eval_inputs)  # type: ignore

    def on_validation_batch_end(  # type: ignore
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Wait for on_validation_batch_end PL hook to call 'process'."""
        if dataloader_idx == self.dataloader_idx:
            self.evaluator.process(batch, outputs)

    def on_sanity_check_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Disable logging of results on sanity check."""
        self.logging_disabled = True
        self.run_eval = False

    def on_sanity_check_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Enable logging of results after sanity check."""
        self.logging_disabled = False
        self.run_eval = True

    def evaluate(self) -> Dict[str, MetricLogs]:
        """Evaluate the performance after processing all input/output pairs."""
        if not self.run_eval:
            return "", {}

        results = {}
        logger = logging.getLogger(__name__)
        if not self.logging_disabled:
            logger.info("Running evaluator %s...", str(self.evaluator))

        for metric in self.evaluator.metrics:
            if self.output_dir is not None:
                output_dir = os.path.join(self.output_dir, metric)
                os.makedirs(output_dir, exist_ok=True)
                self.evaluator.save(output_dir, metric)  # TODO

            log_dict, log_str = self.evaluator.evaluate(metric)
            results[metric] = log_dict
            if not self.logging_disabled:
                for k, v in log_dict.items():
                    self.log(k, v, rank_zero_only=True)  # type: ignore # pylint: disable=no-member,line-too-long
                logger.info("Showing results for %s", metric)
                logger.info(log_str)
        return results
