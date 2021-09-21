"""VisT engine utils."""
import inspect
import itertools
import json
import logging
import os
import sys
import warnings
from argparse import Namespace
from os import path as osp
from typing import Dict, List, Optional, Tuple, no_type_check

import pytorch_lightning as pl
import yaml
from devtools import debug
from pytorch_lightning.callbacks.progress import reset
from pytorch_lightning.utilities import rank_zero_only, rank_zero_info
from scalabel.label.typing import Frame
from termcolor import colored
from torch.utils.collect_env import get_pretty_env_info

from ..common.utils.distributed import (
    all_gather_object_cpu,
    all_gather_object_gpu,
)

# ignore DeprecationWarning by default (e.g. numpy)
from ..config import Config
from ..struct import DictStrAny

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
) -> None:
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


@rank_zero_only
def setup_logging(output_dir: str, trainer_args: DictStrAny, cfg: Config) -> None:
    """Setup command line logger, create output dir, save info."""
    setup_logger(osp.join(output_dir, "log.txt"))

    # print env / config
    rank_zero_info("Environment info: %s", get_pretty_env_info())
    rank_zero_info(
        "Running with full config:\n %s",
        str(debug.format(cfg)).split("\n", 1)[1],
    )
    if cfg.launch.seed is not None:
        rank_zero_info("Using a fixed random seed: %s", cfg.launch.seed)

    # save trainer args (converted to string)
    path = osp.join(output_dir, "trainer_args.yaml")
    for key, arg in trainer_args.items():
        trainer_args[key] = str(arg)
    with open(path, "w", encoding="utf-8") as outfile:
        yaml.dump(trainer_args, outfile, default_flow_style=False)
    rank_zero_info("Trainer arguments saved to %s", path)

    # save VisT config
    path = osp.join(output_dir, "config.json")
    with open(path, "w", encoding="utf-8") as outfile:
        json.dump(trainer_args, outfile)
    rank_zero_info("VisT Config saved to %s", path)


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
