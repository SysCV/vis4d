"""CLI interface for vis4d.

Example to run this script:
>>> python -m vis4d.pl.cli --config vis4d/config/example/faster_rcnn_coco.py
"""
from __future__ import annotations

import logging
import os.path as osp

from absl import app, flags
from ml_collections import ConfigDict
from pytorch_lightning import Callback
from pytorch_lightning.utilities.exceptions import (  # type: ignore[attr-defined] # pylint: disable=line-too-long
    MisconfigurationException,
)
from torch.utils.collect_env import get_pretty_env_info

from vis4d.common.callbacks import VisualizerCallback
from vis4d.common.logging import rank_zero_info, setup_logger
from vis4d.common.util import set_tf32
from vis4d.config.replicator import replicate_config
from vis4d.config.util import instantiate_classes, pprints_config
from vis4d.engine.parser import DEFINE_config_file
from vis4d.pl.callbacks import CallbackWrapper, OptimEpochCallback
from vis4d.pl.data_module import DataModule
from vis4d.pl.trainer import DefaultTrainer
from vis4d.pl.training_module import TrainingModule

_CONFIG = DEFINE_config_file("config", method_name="get_config")
_GPUS = flags.DEFINE_integer("gpus", default=0, help="Number of GPUs")
_CKPT = flags.DEFINE_string("ckpt", default=None, help="Checkpoint path")
_RESUME = flags.DEFINE_bool("resume", default=False, help="Resume training")
_VISUALIZE = flags.DEFINE_bool(
    "visualize", default=False, help="visualize the results"
)
_SWEEP = DEFINE_config_file("sweep", method_name="get_sweep")
_SHOW_CONFIG = flags.DEFINE_bool(
    "print-config", default=False, help="If set, prints the configuration."
)


def main(argv) -> None:  # type:ignore
    """Main entry point for the CLI.

    Example to run this script:
    >>> python -m vis4d.engine.cli --config configs/faster_rcnn/faster_rcnn_coco.py

    To run a parameter sweep:
    >>> python -m vis4d.engine.cli fit --config configs/faster_rcnn/faster_rcnn_coco.py --sweep configs/faster_rcnn/faster_rcnn_coco.py
    """
    # Get config
    mode = argv[1]
    assert mode in {"fit", "test"}, f"Invalid mode: {mode}"
    experiment_config = _CONFIG.value

    if _SWEEP.value is not None:
        # Perform parameter sweep
        rank_zero_info(
            "Found Parameter Sweep in config file. Running Parameter Sweep..."
        )
        experiment_config = _CONFIG.value
        sweep_config = instantiate_classes(_SWEEP.value)

        for run_id, config in enumerate(
            replicate_config(
                experiment_config,
                method=sweep_config.method,
                sampling_args=sweep_config.sampling_args,
                fstring=sweep_config.get("suffix", ""),
            )
        ):
            rank_zero_info(
                f"Running experiment #%d: %s",  # pylint: disable=f-string-without-interpolation, line-too-long
                run_id,
                config.experiment_name,
            )
            run_single_experiment(
                experiment_config,
                mode,
                _GPUS.value,
                _SHOW_CONFIG.value,
                _VISUALIZE.value,
            )
    else:
        # Run single experiment
        run_single_experiment(
            experiment_config,
            mode,
            _GPUS.value,
            _SHOW_CONFIG.value,
            _VISUALIZE.value,
            _CKPT.value,
            _RESUME.value,
        )


def run_single_experiment(
    config: ConfigDict,
    mode: str,
    num_gpus: int = 0,
    show_config: bool = False,
    visualize: bool = False,
    ckpt_path: str | None = None,
    resume: bool = False,
) -> None:
    """Entry point for running a single experiment.

    Args:
        config (ConfigDict): Configuration dictionary.
        mode (str): Mode to run the experiment in. Either `fit` or `test`.
        num_gpus (int): Number of GPUs to use.
        show_config (bool): If set, prints the configuration.
        visualize (bool): If set, activates the visualization callbacks.
        ckpt_path (str): Path to checkpoint to load.
        resume (bool): If set, resumes training from the checkpoint.

    Raises:
        ValueError: If `mode` is not `fit` or `test`.
    """

    # Setup logging
    logger_vis4d = logging.getLogger("vis4d")
    logger_pl = logging.getLogger("pytorch_lightning")
    log_file = osp.join(config.output_dir, f"log_{config.timestamp}.txt")
    setup_logger(logger_vis4d, log_file)
    setup_logger(logger_pl, log_file)

    rank_zero_info("Environment info: %s", get_pretty_env_info())

    # PyTorch Setting
    set_tf32(False)

    # TODO: Update Trainer Args in show config
    if show_config:
        rank_zero_info(pprints_config(config))

    # Setup Trainer kwargs
    trainer_args_cfg = ConfigDict()
    pl_trainer = instantiate_classes(config.pl_trainer)
    for key, value in pl_trainer.items():
        trainer_args_cfg[key] = value

    trainer_args_cfg.work_dir = config.work_dir
    trainer_args_cfg.exp_name = config.experiment_name
    trainer_args_cfg.version = config.version

    if "benchmark" in config:
        trainer_args_cfg.benchmark = config.benchmark
    trainer_args_cfg.num_sanity_val_steps = 0

    # Setup GPU
    trainer_args_cfg.devices = max(1, num_gpus)
    if num_gpus > 0:
        trainer_args_cfg.accelerator = "gpu"

    # Setup logger
    trainer_args_cfg.enable_progress_bar = False
    trainer_args_cfg.log_every_n_steps = 50

    trainer_args = instantiate_classes(trainer_args_cfg)

    # Seed
    seed = config.get("seed", None)

    # Setup sampler
    trainer_args.replace_sampler_ddp = False

    # Instantiate classes
    data_connector = instantiate_classes(config.data_connector)
    optimizers = instantiate_classes(config.optimizers)
    loss = instantiate_classes(config.loss)

    callbacks: list[Callback] = []
    if "shared_callbacks" in config:
        shared_callbacks = instantiate_classes(config.shared_callbacks)
        for key, cb in shared_callbacks.items():
            rank_zero_info(f"Adding callback {key}")
            callbacks.append(CallbackWrapper(cb, data_connector, key))

    if "train_callbacks" in config and mode == "fit":
        train_callbacks = instantiate_classes(config.train_callbacks)
        for key, cb in train_callbacks.items():
            rank_zero_info(f"Adding callback {key}")
            callbacks.append(CallbackWrapper(cb, data_connector, key))

    if "test_callbacks" in config:
        test_callbacks = instantiate_classes(config.test_callbacks)
        for key, cb in test_callbacks.items():
            if isinstance(cb, VisualizerCallback) and not visualize:
                continue
            rank_zero_info(f"Adding callback {key}")
            callbacks.append(CallbackWrapper(cb, data_connector, key))

    if "pl_callbacks" in config:
        pl_callbacks = instantiate_classes(config.pl_callbacks)
    else:
        pl_callbacks = []

    for cb in pl_callbacks:
        if not isinstance(cb, Callback):
            raise MisconfigurationException(
                "Callback must be a subclass of pytorch_lightning.Callback. "
                f"Provided callback: {cb} is not!"
            )
        callbacks.append(cb)

    # Add needed callbacks
    callbacks.append(OptimEpochCallback())

    trainer = DefaultTrainer(callbacks=callbacks, **trainer_args)
    training_module = TrainingModule(
        config.model, optimizers, loss, data_connector, seed
    )
    data_module = DataModule(config.data)

    if resume:
        if ckpt_path is None:
            ckpt_path = osp.join(config.output_dir, "checkpoints/last.ckpt")

    if mode == "fit":
        trainer.fit(
            training_module,
            datamodule=data_module,
            ckpt_path=ckpt_path,
        )
    elif mode == "test":
        trainer.test(
            training_module,
            datamodule=data_module,
            verbose=False,
            ckpt_path=ckpt_path,
        )


if __name__ == "__main__":
    app.run(main)
