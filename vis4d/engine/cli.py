"""CLI interface for vis4d.

Example to run this script:
>>> python -m vis4d.engine.cli --config vis4d/config/example/faster_rcnn_coco.py
"""
from __future__ import annotations

import os

import torch
from absl import app, flags
from ml_collections import ConfigDict
from torch.distributed import destroy_process_group, init_process_group
from torch.multiprocessing import spawn  # type: ignore
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.collect_env import get_pretty_env_info

from vis4d.common.distributed import get_world_size
from vis4d.common.logging import rank_zero_info
from vis4d.config.replicator import replicate_config
from vis4d.config.util import instantiate_classes, pprints_config
from vis4d.engine.parser import DEFINE_config_file
from vis4d.engine.test import Tester
from vis4d.engine.train import Trainer

# Currently this does not allow to load multpile config files.
# Would be nice to extend functionality to chain multiple config files using
# e.g. --config=model_1.py --config=loader_args.py
# or --config=my_config.py --config.train_dl=different_dl.py

_CONFIG = DEFINE_config_file("config", method_name="get_config")
_SWEEP = DEFINE_config_file("sweep", method_name="get_sweep")
_MODE = flags.DEFINE_string(
    "mode", default="train", help="Choice of [train, test]"
)
_GPUS = flags.DEFINE_integer("gpus", default=0, help="Number of GPUs")
_SHOW_CONFIG = flags.DEFINE_bool(
    "print-config", default=False, help="If set, prints the configuration."
)


def _test(config: ConfigDict, rank: None | int = None) -> None:
    """Test the model."""
    # Would be nice to  connect this to SLURM cluster to directly spawn jobs.
    rank_zero_info("Testing model")
    if _SHOW_CONFIG.value:
        rank_zero_info("*" * 80)
        rank_zero_info(pprints_config(config))
        rank_zero_info("*" * 80)

    cfg: ConfigDict = instantiate_classes(config)
    tester = Tester(
        dataloaders=cfg.test_dl,
        data_connector=cfg.data_connector,
        test_callbacks=cfg.get("test_callbacks", None),
    )

    if rank is not None:
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    cfg.model.to(device)
    tester.test(cfg.model)


def _train(config: ConfigDict, rank: None | int = None) -> None:
    """Train the model."""
    cfg: ConfigDict = instantiate_classes(config)

    trainer = Trainer(
        num_epochs=cfg.num_epochs,
        log_step=1,
        dataloaders=cfg.train_dl,
        data_connector=cfg.data_connector,
        train_callbacks=cfg.get("train_callbacks", None),
    )
    tester = Tester(
        dataloaders=cfg.test_dl,
        data_connector=cfg.data_connector,
        test_callbacks=cfg.get("test_callbacks", None),
    )

    if rank is not None:
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")
    cfg.model.to(device)
    if get_world_size() > 1:
        assert rank is not None, "Requires rank for multi-processing"
        cfg.model = DDP(cfg.model, device_ids=[rank])

    # run training
    trainer.train(cfg.model, cfg.optimizers, cfg.loss, tester)


def ddp_setup(rank: int, world_size: int) -> None:
    """Setup DDP environment and init processes.

    Args:
        rank (int): Unique identifier of each process.
        world_size (int): Total number of processes.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["LOCAL_RANK"] = str(rank)
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def _dist_train(rank: int, world_size: int, config: ConfigDict) -> None:
    """Train script setting up DDP, executing action, terminating."""
    ddp_setup(rank, world_size)
    _train(config, rank)
    destroy_process_group()


def train(config: ConfigDict) -> None:
    """Train the model. If multiple GPUs are available, uses DDP."""
    rank_zero_info("Starting training")
    rank_zero_info("Environment info: %s", get_pretty_env_info())

    # Would be nice to  connect this to SLURM cluster to directly spawn jobs.)
    if _SHOW_CONFIG.value:
        rank_zero_info("*" * 80)
        rank_zero_info(pprints_config(config))
        rank_zero_info("*" * 80)

    if torch.cuda.is_available():
        rank_zero_info(
            "\n Using %d/%d GPUs", config.n_gpus, torch.cuda.device_count()
        )
    if config.n_gpus > 1:
        spawn(_dist_train, args=(config.n_gpus, config), nprocs=config.n_gpus)
    else:
        _train(config, 0 if config.n_gpus == 1 else None)


def main(  # type:ignore # pylint: disable=unused-argument
    *args, **kwargs
) -> None:
    """Main entry point for the CLI.

    Example to run this script:
    >>> python -m vis4d.engine.cli --config vis4d/config/example/faster_rcnn_coco.py

    Or to run a parameter sweep:
    >>> python -m vis4d.engine.cli --config vis4d/config/example/faster_rcnn_coco.py --sweep vis4d/config/example/faster_rcnn_coco.py
    """
    config = _CONFIG.value

    if _GPUS.value != config.n_gpus:
        # Replace # gpus in config with cli
        config.n_gpus = _GPUS.value  # patch it like this to update potential
        # references to the config
    num_gpus = config.n_gpus

    if _SWEEP.value is not None:
        # Perform parameter sweep
        # TODO, improve. Where to save the results? What name for the run?
        config = _CONFIG.value
        sweep_obj = instantiate_classes(_SWEEP.value)

        for config in replicate_config(
            _CONFIG.value,
            method=sweep_obj.method,
            sampling_args=sweep_obj.sampling_args,
        ):
            train(_CONFIG.value)

    elif _MODE.value == "train":
        train(_CONFIG.value)

    elif _MODE.value == "test":
        _test(_CONFIG.value, 0 if num_gpus > 0 else None)


if __name__ == "__main__":
    app.run(main)
