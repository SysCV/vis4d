"""Vis4D engine utils."""
import datetime
import inspect
import itertools
import json
import logging
import os
import sys
import warnings
from argparse import Namespace
from os import path as osp
from typing import Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import yaml
from devtools import debug
from pytorch_lightning.callbacks.progress.tqdm_progress import reset
from pytorch_lightning.utilities.distributed import (
    rank_zero_info,
    rank_zero_only,
)
from scalabel.label.typing import Frame
from termcolor import colored
from torch.utils.collect_env import get_pretty_env_info

from ..common.utils.distributed import (
    all_gather_object_cpu,
    all_gather_object_gpu,
)
from ..common.utils.time import Timer
from ..config import Config
from ..struct import DictStrAny, InputSample, LossesType, ModelOutput

try:
    from mmcv.utils import get_logger

    MMCV_INSTALLED = True
except (ImportError, NameError):  # pragma: no cover
    MMCV_INSTALLED = False

logger = logging.getLogger("pytorch_lightning")
# ignore DeprecationWarning by default (e.g. numpy)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class Vis4DTQDMProgressBar(pl.callbacks.TQDMProgressBar):  # type: ignore
    """TQDMProgressBar keeping training and validation progress separate."""

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Reset progress bar using total training batches."""
        self._train_batch_idx = 0
        reset(self.main_progress_bar, self.total_train_batches)
        self.main_progress_bar.set_description(
            f"Epoch {trainer.current_epoch + 1}"
        )


class Vis4DProgressBar(pl.callbacks.ProgressBarBase):  # type: ignore
    """ProgressBar with separate printout per log step."""

    def __init__(self, refresh_rate: int = 50) -> None:
        """Init."""
        super().__init__()
        self._refresh_rate = refresh_rate
        self.enable = True
        self.timer = Timer()
        self._metrics_history: List[DictStrAny] = []

    def disable(self) -> None:
        """Disable progressbar."""
        self.enable = False  # pragma: no cover

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Reset timer on start of epoch."""
        super().on_train_epoch_start(trainer, pl_module)
        self.timer.reset()
        self._metrics_history = []

    def on_predict_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Reset timer on start of epoch."""
        super().on_train_epoch_start(trainer, pl_module)
        self.timer.reset()
        self._metrics_history = []

    def on_validation_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Reset timer on start of validation."""
        super().on_train_epoch_start(trainer, pl_module)
        self.timer.reset()
        self._metrics_history = []

    def on_test_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Reset timer on start of test."""
        super().on_train_epoch_start(trainer, pl_module)
        self.timer.reset()
        self._metrics_history = []

    def _get_metrics(self) -> DictStrAny:
        """Get current running avg of metrics and clean history."""
        acc_metrics = {}
        for k, v in self._metrics_history[-1].items():
            if isinstance(v, (torch.Tensor, float, int)):
                acc_value = 0.0
                num_hist = 0
                for hist_dict in self._metrics_history:
                    if k in hist_dict:
                        acc_value += hist_dict[k]
                        num_hist += 1
                acc_value /= num_hist
                acc_metrics[k] = acc_value
            elif isinstance(v, str) and not v == "nan":
                acc_metrics[k] = v
        self._metrics_history = []
        return acc_metrics

    def _compose_log_str(
        self,
        prefix: str,
        batch_idx: int,
        total_batches: int,
    ) -> str:
        """Compose log str from given information."""
        time_sec_tot = self.timer.time()
        time_sec_avg = time_sec_tot / (batch_idx + 1)
        eta_sec = time_sec_avg * (total_batches - (batch_idx + 1))
        if not eta_sec == float("inf"):
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
        else:  # pragma: no cover
            eta_str = "---"

        metrics_list = []
        for k, v in self._get_metrics().items():
            if isinstance(v, (torch.Tensor, float)):
                kv_str = f"{k}: {v:.3f}"
            else:
                kv_str = f"{k}: {v}"
            metrics_list.append(kv_str)
        metr_str = ", ".join(metrics_list)
        time_str = f"ETA: {eta_str}, " + (
            f"{time_sec_avg:.2f}s/it"
            if time_sec_avg > 1
            else f"{1/time_sec_avg:.2f}it/s"
        )
        logging_str = (
            f"{prefix}: {batch_idx}/{total_batches}, {time_str}, {metr_str}"
        )
        return logging_str

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: LossesType,
        batch: List[InputSample],
        batch_idx: int,
    ) -> None:
        """Train phase logging."""
        super().on_train_batch_end(
            trainer, pl_module, outputs, batch, batch_idx
        )
        metrics = self.get_metrics(trainer, pl_module)
        self._metrics_history.append(metrics)

        if batch_idx % self._refresh_rate == 0 and self.enable:
            rank_zero_info(
                self._compose_log_str(
                    f"Epoch {trainer.current_epoch + 1}",
                    batch_idx,
                    self.total_train_batches,
                )
            )

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: ModelOutput,
        batch: List[InputSample],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Validation phase logging."""
        super().on_validation_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )
        metrics = self.get_metrics(trainer, pl_module)
        self._metrics_history.append(metrics)

        if batch_idx % self._refresh_rate == 0 and self.enable:
            rank_zero_info(
                self._compose_log_str(
                    "Validating",
                    batch_idx,
                    self.total_val_batches,
                )
            )

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: ModelOutput,
        batch: List[InputSample],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Test phase logging."""
        super().on_test_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )
        metrics = self.get_metrics(trainer, pl_module)
        self._metrics_history.append(metrics)

        if batch_idx % self._refresh_rate == 0 and self.enable:
            rank_zero_info(
                self._compose_log_str(
                    "Testing",
                    batch_idx,
                    self.total_test_batches,
                )
            )

    def on_predict_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: ModelOutput,
        batch: List[InputSample],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Predict phase logging."""
        super().on_predict_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )
        metrics = self.get_metrics(trainer, pl_module)
        self._metrics_history.append(metrics)

        if batch_idx % self._refresh_rate == 0 and self.enable:
            rank_zero_info(
                self._compose_log_str(
                    "Predicting",
                    batch_idx,
                    self.total_predict_batches,
                )
            )


class _ColorFormatter(logging.Formatter):
    """Formatter for terminal messages with colors."""

    def formatMessage(self, record: logging.LogRecord) -> str:
        """Add appropriate color to log message."""
        log = super().formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno in [logging.ERROR, logging.CRITICAL]:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


def setup_logger(
    filepath: Optional[str] = None,
    color: bool = True,
    std_out_level: int = logging.DEBUG,
) -> None:
    """Configure logging for Vis4D using the pytorch lightning logger."""
    # get PL logger, remove handlers to re-define behavior
    # https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html#configure-console-logging
    for h in logger.handlers:
        logger.removeHandler(h)

    # console logger
    plain_formatter = logging.Formatter(
        "[%(asctime)s] Vis4D %(levelname)s: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(std_out_level)
    if color:
        formatter = _ColorFormatter(
            colored("[%(asctime)s Vis4D]: ", "green") + "%(message)s",
            datefmt="%m/%d %H:%M:%S",
        )
        ch.setFormatter(formatter)
    else:
        ch.setFormatter(plain_formatter)
    logger.addHandler(ch)

    # file logger
    if filepath is not None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fh = logging.FileHandler(filepath)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)

    if MMCV_INSTALLED:
        mm_logger = get_logger("mmdet")
        mm_logger.setLevel(logging.ERROR)
        mm_logger = get_logger("mmcv")
        mm_logger.setLevel(logging.ERROR)


@rank_zero_only
def setup_logging(
    output_dir: str, trainer_args: DictStrAny, cfg: Config
) -> None:
    """Setup command line logger, create output dir, save info."""
    setup_logger(osp.join(output_dir, "log.txt"))

    # print env / config
    rank_zero_info("Environment info: %s", get_pretty_env_info())
    rank_zero_info(
        "Running with full config:\n %s",
        str(debug.format(cfg)).split("\n", 1)[1],
    )
    if cfg.launch.seed is not None:
        rank_zero_info("Using random seed: %s", cfg.launch.seed)

    # save trainer args (converted to string)
    path = osp.join(output_dir, "trainer_args.yaml")
    for key, arg in trainer_args.items():
        trainer_args[key] = str(arg)
    with open(path, "w", encoding="utf-8") as outfile:
        yaml.dump(trainer_args, outfile, default_flow_style=False)
    rank_zero_info("Trainer arguments saved to %s", path)

    # save Vis4D config
    path = osp.join(output_dir, "config.json")
    with open(path, "w", encoding="utf-8") as outfile:
        json.dump(trainer_args, outfile)
    rank_zero_info("Vis4D Config saved to %s", path)


def split_args(args: Namespace) -> Tuple[Namespace, DictStrAny]:
    """Split argparse Namespace into Vis4D and pl.Trainer arguments."""
    params = vars(args)
    valid_kwargs = inspect.signature(pl.Trainer.__init__).parameters
    trainer_kwargs = Namespace(
        **{name: params[name] for name in valid_kwargs if name in params}
    )
    vis4d_kwargs = Namespace(
        **{name: params[name] for name in params if name not in valid_kwargs}
    )
    return vis4d_kwargs, vars(trainer_kwargs)


def all_gather_predictions(
    predictions: Dict[str, List[Frame]],
    pl_module: pl.LightningModule,
    collect_device: str,
) -> Optional[Dict[str, List[Frame]]]:  # pragma: no cover
    """Gather prediction dict in distributed setting."""
    if collect_device == "gpu":
        predictions_list = all_gather_object_gpu(predictions, pl_module)
    elif collect_device == "cpu":
        predictions_list = all_gather_object_cpu(predictions, pl_module)
    else:
        raise ValueError(f"Collect device {collect_device} unknown.")

    if predictions_list is None:
        return None

    result = {}
    for key in predictions:
        prediction_list = [p[key] for p in predictions_list]
        result[key] = list(itertools.chain(*prediction_list))
    return result


def all_gather_gts(
    gts: List[Frame], pl_module: pl.LightningModule, collect_device: str
) -> Optional[List[Frame]]:  # pragma: no cover
    """Gather gts list in distributed setting."""
    if collect_device == "gpu":
        gts_list = all_gather_object_gpu(gts, pl_module)
    elif collect_device == "cpu":
        gts_list = all_gather_object_cpu(gts, pl_module)
    else:
        raise ValueError(f"Collect device {collect_device} unknown.")

    if gts_list is None:
        return None

    return list(itertools.chain(*gts_list))
