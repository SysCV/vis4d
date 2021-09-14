"""VisT engine utils."""
import inspect
import itertools
import logging
import os
import sys
import warnings
from argparse import Namespace
from typing import Dict, List, Optional, Tuple, no_type_check

import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import reset
from scalabel.label.typing import Frame
from termcolor import colored

from ..common.utils.distributed import (
    all_gather_object_cpu,
    all_gather_object_gpu,
)

# ignore DeprecationWarning by default (e.g. numpy)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class VisTProgressBar(pl.callbacks.ProgressBar):
    """ProgressBar keeping training and validation progress separate."""

    @no_type_check
    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Reset progress bar using total training batches."""
        self._train_batch_idx = 0
        reset(self.main_progress_bar, self.total_train_batches)
        self.main_progress_bar.set_description(
            f"Epoch {trainer.current_epoch}"
        )


class _ColorFormatter(logging.Formatter):
    """Formatter for terminal messages with colors."""

    def formatMessage(self, record: logging.LogRecord) -> str:
        """Add appropriate color to log message."""
        log = super().formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif (
            record.levelno == logging.ERROR
            or record.levelno == logging.CRITICAL
        ):
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


def setup_logger(
    filepath: Optional[str] = None,
    color: bool = True,
    std_out_level: int = logging.DEBUG,
) -> logging.Logger:
    """Configure logging for VisT using the pytorch lightning logger."""
    # get PL logger, remove handlers to re-define behavior
    # https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html#configure-console-logging
    logger = logging.getLogger("pytorch_lightning")
    for h in logger.handlers:
        logger.removeHandler(h)

    # console logger
    plain_formatter = logging.Formatter(
        "[%(asctime)s] VisT %(levelname)s: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(std_out_level)
    if color:
        formatter = _ColorFormatter(
            colored("[%(asctime)s VisT]: ", "green") + "%(message)s",
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
    return logger


def split_args(args: Namespace) -> Tuple[Namespace, Namespace]:
    """Split argparse Namespace into VisT and pl.Trainer arguments."""
    params = vars(args)
    valid_kwargs = inspect.signature(pl.Trainer.__init__).parameters
    trainer_kwargs = Namespace(
        **{name: params[name] for name in valid_kwargs if name in params}
    )
    vist_kwargs = Namespace(
        **{name: params[name] for name in params if name not in valid_kwargs}
    )
    return vist_kwargs, trainer_kwargs


def all_gather_predictions(
    predictions: Dict[str, List[Frame]],
    pl_module: pl.LightningModule,
    collect_fn: str,
) -> Optional[Dict[str, List[Frame]]]:
    """Gather prediction dict in distributed setting."""
    if collect_fn == "gpu":
        predictions_list = all_gather_object_gpu(predictions, pl_module)
    elif collect_fn == "cpu":
        predictions_list = all_gather_object_cpu(predictions)
    else:
        raise ValueError(f"Collect arg {collect_fn} unknown.")

    if predictions_list is None:
        return None

    result = {}
    for key in predictions:
        prediction_list = [p[key] for p in predictions_list]
        result[key] = list(itertools.chain(*prediction_list))
    return result


def all_gather_gts(
    gts: List[Frame], pl_module: pl.LightningModule, collect_fn: str
) -> Optional[List[Frame]]:
    """Gather gts list in distributed setting."""
    if collect_fn == "gpu":
        gts_list = all_gather_object_gpu(gts, pl_module)
    elif collect_fn == "cpu":
        gts_list = all_gather_object_cpu(gts)
    else:
        raise ValueError(f"Collect arg {collect_fn} unknown.")

    if gts_list is None:
        return None

    return list(itertools.chain(*gts_list))
