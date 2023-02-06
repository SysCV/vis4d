"""Evaluation components for tracking."""
from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT

from vis4d.common.callbacks import EvaluatorCallback
from vis4d.engine.connectors import DataConnector, StaticDataConnector
from vis4d.eval.base import Evaluator

# from vis4d.pl.distributed import all_gather_object_cpu


class DefaultEvaluatorCallback(Callback):
    """Base class for Vis4D Evaluators.

    This class will accumulate the inputs/outputs in 'process', and produce
    evaluation results in 'evaluate'.
    """

    def __init__(
        self,
        dataloader_idx: int,
        evaluator: Evaluator,
        eval_connector: DataConnector = StaticDataConnector(
            {"train": {}, "test": {}, "loss": {}}  # TODO: better default value
        ),
        output_dir: None | str = None,
        collect: str = "cpu",
    ) -> None:
        """Init class."""
        self.dataloader_idx = dataloader_idx
        self.eval_callback = EvaluatorCallback(
            evaluator, eval_connector, output_dir=output_dir, collect=collect
        )
        self.logging_disabled = False
        self.run_eval = True

    def on_test_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Wait for on_test_epoch_end PL hook to call 'evaluate'."""
        self.eval_callback.on_test_epoch_end()

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Wait for on_validation_epoch_end PL hook to call 'evaluate'."""
        self.eval_callback.on_test_epoch_end()

    def on_test_batch_end(  # type: ignore
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Wait for on_test_batch_end PL hook to call 'process'."""
        if dataloader_idx == self.dataloader_idx:
            self.eval_callback.on_test_batch_end(outputs, batch, "")

    def on_validation_batch_end(  # type: ignore
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Wait for on_validation_batch_end PL hook to call 'process'."""
        if dataloader_idx == self.dataloader_idx:
            self.eval_callback.on_test_batch_end(outputs, batch, "")

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
