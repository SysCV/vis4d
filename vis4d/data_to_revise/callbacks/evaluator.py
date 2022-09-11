"""Evaluation components for tracking."""
import copy
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from scalabel.common import mute
from scalabel.label.typing import Frame

from vis4d.data_to_clean.datasets import BaseDatasetLoader
from vis4d.data_to_clean.utils import all_gather_gts, all_gather_predictions
from vis4d.struct import InputSample, MetricLogs, ModelOutput

mute(True)  # turn off undesired logs during eval
logger = logging.getLogger("pytorch_lightning")


class BaseEvaluatorCallback(Callback):
    """Base class for Vis4D Evaluators.

    This class will accumulate the inputs/outputs in 'process', and produce
    evaluation results in 'evaluate'.
    """

    def __init__(self, dataloader_idx: int, collect: str = "cpu") -> None:
        """Init class."""
        assert collect in ["cpu", "gpu"], f"Collect arg {collect} unknown."
        self._predictions: Dict[str, List[Frame]] = defaultdict(list)
        self._gts: List[Frame] = []
        self.logging_disabled = False
        self.collect = collect
        self.dataloader_idx = dataloader_idx

    def reset(self) -> None:
        """Preparation for a new round of evaluation."""
        self._predictions = defaultdict(list)
        self._gts = []

    def gather(self, pl_module: pl.LightningModule) -> None:
        """Gather accumulated data."""
        preds = all_gather_predictions(
            self._predictions, pl_module, self.collect
        )
        if preds is not None:
            self._predictions = preds
        gts = all_gather_gts(self._gts, pl_module, self.collect)
        if gts is not None:
            self._gts = gts

    def process(
        self, inputs: List[List[InputSample]], outputs: ModelOutput
    ) -> None:
        """Process the pair of inputs and outputs."""
        raise NotImplementedError

    def evaluate(self, epoch: int) -> Dict[str, MetricLogs]:
        """Evaluate the performance after processing all input/output pairs."""
        raise NotImplementedError

    def on_test_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Wait for on_test_epoch_end PL hook to call 'evaluate'."""
        self.gather(pl_module)
        if trainer.is_global_zero:
            self.evaluate(trainer.current_epoch)
        self.reset()

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Wait for on_validation_epoch_end PL hook to call 'evaluate'."""
        self.gather(pl_module)
        if trainer.is_global_zero:
            self.evaluate(trainer.current_epoch)
        self.reset()

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
            self.process(batch, outputs)  # type: ignore

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
            self.process(batch, outputs)  # type: ignore

    def on_sanity_check_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Disable logging of results on sanity check."""
        self.logging_disabled = True

    def on_sanity_check_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Enable logging of results after sanity check."""
        self.logging_disabled = False


class DefaultEvaluatorCallback(BaseEvaluatorCallback):
    """Evaluate model using metrics supported by the dataset."""

    def __init__(
        self,
        dataloader_idx: int,
        dataset_loader: BaseDatasetLoader,
        output_dir: Optional[str] = None,
    ) -> None:
        """Init."""
        super().__init__(dataloader_idx, dataset_loader.collect_device)
        self.output_dir = output_dir
        self.name = dataset_loader.name
        self.metrics = dataset_loader.eval_metrics
        self.eval_func = dataset_loader.evaluate
        self.save_func = dataset_loader.save_predictions

        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)

    def process(
        self, inputs: List[List[InputSample]], outputs: ModelOutput
    ) -> None:
        """Process the pair of inputs and outputs."""
        for inp in inputs:
            self._gts.append(copy.deepcopy(inp[0].metadata[0]))

        for key, output in outputs.items():
            for inp, out in zip(inputs, output):
                prediction = copy.deepcopy(inp[0].metadata[0])
                prediction.labels = out
                self._predictions[key].append(prediction)

    def evaluate(self, epoch: int) -> Dict[str, MetricLogs]:
        """Evaluate the performance after processing all input/output pairs."""
        results = {}
        assert self.metrics is not None
        if not self.logging_disabled and len(self.metrics) > 0:
            logger.info("Running evaluation on dataset %s...", self.name)
        for key, predictions in self._predictions.items():
            if self.output_dir is not None:
                output_dir = os.path.join(self.output_dir, key)
                os.makedirs(output_dir, exist_ok=True)
                self.save_func(output_dir, key, predictions)

            if key in self.metrics:
                log_dict, log_str = self.eval_func(key, predictions, self._gts)
                results[key] = log_dict
                if not self.logging_disabled:
                    for k, v in log_dict.items():
                        self.log(k, v, rank_zero_only=True)  # type: ignore # pylint: disable=no-member,line-too-long
                    logger.info("Showing results for %s", key)
                    logger.info(log_str)
        return results
