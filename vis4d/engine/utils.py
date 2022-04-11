"""Vis4D engine utils."""
import datetime
import logging
import os
import sys
import warnings
from typing import List, Optional, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.rank_zero import (
    rank_zero_info,
    rank_zero_only,
)
from pytorch_lightning.utilities.types import STEP_OUTPUT
from termcolor import colored

from ..common.utils.time import Timer
from ..struct import ArgsType, DictStrAny

try:
    from mmcv.utils import get_logger

    mm_logger = get_logger("mmdet")
    mm_logger.setLevel(logging.WARNING)

    MMCV_INSTALLED = True
except (ImportError, NameError):  # pragma: no cover
    MMCV_INSTALLED = False

logger = logging.getLogger("pytorch_lightning")
# ignore DeprecationWarning by default (e.g. numpy)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class DefaultProgressBar(pl.callbacks.ProgressBarBase):  # type: ignore
    """ProgressBar with separate printout per log step."""

    def __init__(self, refresh_rate: int = 50) -> None:
        """Init."""
        super().__init__()
        self._refresh_rate = refresh_rate
        self.enable()
        self.timer = Timer()
        self._metrics_history: List[DictStrAny] = []

    def disable(self) -> None:
        """Disable progressbar."""
        self._enabled = False  # pragma: no cover

    def enable(self) -> None:
        """Enable progressbar."""
        self._enabled = True

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Reset timer on start of epoch."""
        super().on_train_epoch_start(trainer, pl_module)
        self.timer.reset()
        self._metrics_history = []

    def on_predict_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Reset timer on start of predict."""
        super().on_predict_start(trainer, pl_module)
        self.timer.reset()
        self._metrics_history = []

    def on_validation_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Reset timer on start of validation."""
        super().on_validation_start(trainer, pl_module)
        self.timer.reset()
        self._metrics_history = []

    def on_test_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Reset timer on start of test."""
        super().on_test_start(trainer, pl_module)
        self.timer.reset()
        self._metrics_history = []

    def _get_metrics(self) -> DictStrAny:
        """Get current running avg of metrics and clean history."""
        acc_metrics: DictStrAny = {}
        if len(self._metrics_history) == 0:
            return acc_metrics

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
        total_batches: Union[int, float],
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

        time_str = f"ETA: {eta_str}, " + (
            f"{time_sec_avg:.2f}s/it"
            if time_sec_avg > 1
            else f"{1/time_sec_avg:.2f}it/s"
        )
        logging_str = f"{prefix}: {batch_idx}/{total_batches}, {time_str}"
        if len(metrics_list) > 0:
            logging_str += ", " + ", ".join(metrics_list)
        return logging_str

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: ArgsType,
        batch_idx: int,
        unused: int = 0,
    ) -> None:
        """Train phase logging."""
        super().on_train_batch_end(
            trainer, pl_module, outputs, batch, batch_idx
        )
        metrics = self.get_metrics(trainer, pl_module)
        self._metrics_history.append(metrics)

        if (
            self.train_batch_idx - 1
        ) % self._refresh_rate == 0 and self._enabled:
            rank_zero_info(
                self._compose_log_str(
                    f"Epoch {trainer.current_epoch}",
                    self.train_batch_idx,
                    self.total_train_batches,
                )
            )

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: ArgsType,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Validation phase logging."""
        super().on_validation_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )

        if (
            self.val_batch_idx - 1
        ) % self._refresh_rate == 0 and self._enabled:
            rank_zero_info(
                self._compose_log_str(
                    "Validating",
                    self.val_batch_idx,
                    self.trainer.num_val_batches[dataloader_idx],
                )
            )

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: ArgsType,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Test phase logging."""
        super().on_test_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )

        if (
            self.test_batch_idx - 1
        ) % self._refresh_rate == 0 and self._enabled:
            rank_zero_info(
                self._compose_log_str(
                    "Testing",
                    self.train_batch_idx,
                    self.trainer.num_test_batches[dataloader_idx],
                )
            )

    def on_predict_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: ArgsType,
        batch: ArgsType,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Predict phase logging."""
        super().on_predict_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )

        if (
            self.predict_batch_idx - 1
        ) % self._refresh_rate == 0 and self._enabled:
            rank_zero_info(
                self._compose_log_str(
                    "Predicting",
                    self.predict_batch_idx,
                    self.trainer.num_predict_batches[dataloader_idx],
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


@rank_zero_only
def setup_logger(
    filepath: Optional[str] = None,
    color: bool = True,
    std_out_level: int = logging.INFO,
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
