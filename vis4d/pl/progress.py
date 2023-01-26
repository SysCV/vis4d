# pylint: disable=consider-using-alias,consider-alternative-union-syntax
"""Progress bar utils."""

import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT

from vis4d.common.logging import rank_zero_info

from ..common import ArgsType
from ..common.time import Timer


class DefaultProgressBar(pl.callbacks.ProgressBarBase):  # type: ignore
    """ProgressBar with separate printout per log step."""

    def __init__(self, refresh_rate: int = 50) -> None:
        """Creates an instance of the class."""
        super().__init__()
        self._refresh_rate = refresh_rate
        self._enabled = True
        self._metrics: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self.timer = Timer()

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
        self._metrics = defaultdict(list)

    def on_predict_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Reset timer on start of predict."""
        super().on_predict_start(trainer, pl_module)
        self.timer.reset()

    def on_validation_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Reset timer on start of validation."""
        super().on_validation_start(trainer, pl_module)
        self.timer.reset()

    def on_test_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Reset timer on start of test."""
        super().on_test_start(trainer, pl_module)
        self.timer.reset()

    def _compose_log_str(
        self,
        prefix: str,
        batch_idx: int,
        total_batches: Union[int, float],
        metrics: Optional[Dict[str, Union[int, float, torch.Tensor]]] = None,
    ) -> str:
        """Compose log str from given information."""
        time_sec_tot = self.timer.time()
        time_sec_avg = time_sec_tot / batch_idx
        eta_sec = time_sec_avg * (total_batches - batch_idx)
        if not eta_sec == float("inf"):
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
        else:  # pragma: no cover
            eta_str = "---"

        metrics_list: List[str] = []
        if metrics is not None:
            for k, v in metrics.items():
                name = k.split("/")[-1]  # remove prefix, e.g. train/loss
                if isinstance(v, (torch.Tensor, float)):
                    kv_str = (
                        f"{name}: {v:.3f}"
                        if isinstance(v, (torch.Tensor, float))
                        else f"{name}: {v}"
                    )
                if name == "loss":  # put total loss first
                    metrics_list.insert(0, kv_str)
                else:
                    metrics_list.append(kv_str)

        time_str = f"ETA: {eta_str}, " + (
            f"{time_sec_avg:.2f}s/it"
            if time_sec_avg > 1
            else f"{1/time_sec_avg:.2f}it/s"
        )
        logging_str = f"{prefix}: {batch_idx - 1}/{total_batches}, {time_str}"
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
    ) -> None:
        """Train phase logging."""
        super().on_train_batch_end(
            trainer, pl_module, outputs, batch, batch_idx
        )

        if self._enabled:
            for k, v in trainer.callback_metrics.items():
                self._metrics[k].append(v)
            if (self.train_batch_idx - 1) % self._refresh_rate == 0:
                rank_zero_info(
                    self._compose_log_str(
                        f"Epoch {trainer.current_epoch}",
                        self.train_batch_idx,
                        self.total_train_batches,
                        {
                            k: sum(v) / len(v) if len(v) > 0 else float("NaN")
                            for k, v in self._metrics.items()
                            if k in trainer.progress_bar_metrics.keys()
                        },
                    )
                )
                self._metrics = defaultdict(list)

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
                    self.test_batch_idx,
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
