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
from scalabel.eval.detect import evaluate_det
from scalabel.eval.ins_seg import evaluate_ins_seg
from scalabel.eval.mot import acc_single_video_mot, evaluate_track
from scalabel.eval.mots import acc_single_video_mots, evaluate_seg_track
from scalabel.eval.result import Result
from scalabel.eval.sem_seg import evaluate_sem_seg
from scalabel.label.io import group_and_sort, save
from scalabel.label.typing import Config, Frame

from ..data.datasets import BaseDatasetLoader
from ..struct import InputSample, ModelOutput
from .utils import all_gather_gts, all_gather_predictions

mute(True)  # turn off undesired logs during eval
logger = logging.getLogger("pytorch_lightning")


def _detect(
    pred: List[Frame],
    gt: List[Frame],
    cfg: Config,
    ignore_unknown_cats: bool,  # pylint: disable=unused-argument
) -> Result:
    """Wrapper for evaluate_det function."""
    return evaluate_det(gt, pred, cfg, nproc=1)


def _ins_seg(
    pred: List[Frame],
    gt: List[Frame],
    cfg: Config,
    ignore_unknown_cats: bool,  # pylint: disable=unused-argument
) -> Result:
    """Wrapper for evaluate_ins_seg function."""
    return evaluate_ins_seg(gt, pred, cfg, nproc=1)


def _track(
    pred: List[Frame], gt: List[Frame], cfg: Config, ignore_unknown_cats: bool
) -> Result:
    """Wrapper for evaluate_track function."""
    return evaluate_track(
        acc_single_video_mot,
        group_and_sort(gt),
        group_and_sort(pred),
        cfg,
        nproc=1,
        ignore_unknown_cats=ignore_unknown_cats,
    )


def _seg_track(
    pred: List[Frame], gt: List[Frame], cfg: Config, ignore_unknown_cats: bool
) -> Result:
    """Wrapper for evaluate_seg_track function."""
    return evaluate_seg_track(
        acc_single_video_mots,
        group_and_sort(gt),
        group_and_sort(pred),
        cfg,
        nproc=1,
        ignore_unknown_cats=ignore_unknown_cats,
    )


def _sem_seg(
    pred: List[Frame],
    gt: List[Frame],
    cfg: Config,
    ignore_unknown_cats: bool,  # pylint: disable=unused-argument
) -> Result:
    """Wrapper for evaluate_sem_seg function."""
    return evaluate_sem_seg(gt, pred, cfg, nproc=1)


_eval_mapping = dict(
    detect=_detect,
    track=_track,
    ins_seg=_ins_seg,
    seg_track=_seg_track,
    sem_seg=_sem_seg,
)


class Vis4DEvaluatorCallback(Callback):
    """Base class for Vis4D Evaluators.

    This class will accumulate the inputs/outputs in 'process', and produce
    evaluation results in 'evaluate'.
    """

    def __init__(self, dataloader_idx: int, collect: str = "cpu") -> None:
        """Init class."""
        assert collect in ["cpu", "gpu"], f"Collect arg {collect} unknown."
        self._predictions: Dict[str, List[Frame]] = defaultdict(list)
        self._gts: List[Frame] = []
        self.logger: Optional[pl.loggers.LightningLoggerBase] = None
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

    def evaluate(self, epoch: int) -> Dict[str, Result]:
        """Evaluate the performance after processing all input/output pairs."""
        raise NotImplementedError

    def setup(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: Optional[str] = None,
    ) -> None:
        """Setup logging of results."""
        self.logger = trainer.logger

    def on_test_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
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


class ScalabelEvaluatorCallback(Vis4DEvaluatorCallback):
    """Evaluate model using metrics supported by Scalabel."""

    def __init__(
        self,
        dataloader_idx: int,
        dataset_loader: BaseDatasetLoader,
        output_dir: Optional[str] = None,
    ) -> None:
        """Init."""
        super().__init__(dataloader_idx, dataset_loader.cfg.collect_device)
        self.output_dir = output_dir
        self.ignore_unknown_cats = dataset_loader.cfg.ignore_unkown_cats
        self.name = dataset_loader.cfg.name
        self.dataset_config = dataset_loader.metadata_cfg

        for metric in dataset_loader.cfg.eval_metrics:
            if metric not in _eval_mapping.keys():  # pragma: no cover
                raise KeyError(f"metric {metric} is not supported")
        self.metrics = dataset_loader.cfg.eval_metrics

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

    def evaluate(self, epoch: int) -> Dict[str, Result]:
        """Evaluate the performance after processing all input/output pairs."""
        results = {}
        if not self.logging_disabled and len(self.metrics) > 0:
            logger.info("Running evaluation on dataset %s...", self.name)
        for key, predictions in self._predictions.items():
            if self.output_dir:
                os.makedirs(self.output_dir, exist_ok=True)
                file_path = os.path.join(
                    self.output_dir, f"{key}_predictions.json"
                )
                save(file_path, predictions)

            if key in self.metrics:
                results[key] = _eval_mapping[key](
                    predictions,
                    self._gts,
                    self.dataset_config,
                    self.ignore_unknown_cats,
                )
                if not self.logging_disabled:
                    if self.logger is not None:
                        log_dict = {
                            f"{key}/{metric}": value
                            for metric, value in results[key].summary().items()
                        }
                        self.logger.log_metrics(log_dict, epoch)
                    logger.info("Showing results for %s", key)
                    logger.info(results[key])
        return results
