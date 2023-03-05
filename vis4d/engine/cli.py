"""CLI interface for vis4d.

Example to run this script:
>>> python -m vis4d.engine.cli --config vis4d/config/example/faster_rcnn_coco.py
"""
from __future__ import annotations

import logging
import os

import torch
from absl import app, flags
from ml_collections import ConfigDict
from torch.distributed import destroy_process_group, init_process_group
from torch.multiprocessing import spawn  # type: ignore
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.collect_env import get_pretty_env_info

from vis4d.common.distributed import get_world_size, get_rank, get_local_rank
from vis4d.common.logging import rank_zero_info, setup_logger, _info
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


def ddp_setup(
    torch_distributed_backend: str = "nccl",
) -> None:
    """Setup DDP environment and init processes.

    Args:
        torch_distributed_backend: Backend to use (includes `nccl` and `gloo`)
    """
    global_rank = get_rank()
    world_size = get_world_size()
    _info(
        f"Initializing distributed: "
        f"GLOBAL_RANK: {global_rank}, MEMBER: {global_rank + 1}/{world_size}"
    )
    init_process_group(
        torch_distributed_backend,
        rank=global_rank,
        world_size=world_size,
    )

    # On rank=0 let everyone know training is starting
    rank_zero_info(
        f"{'-' * 100}\n"
        f"distributed_backend={torch_distributed_backend}\n"
        f"All distributed processes registered. "
        f"Starting with {world_size} processes\n"
        f"{'-' * 100}\n"
    )

    local_rank = get_local_rank()
    all_gpu_ids = ",".join(str(x) for x in range(torch.cuda.device_count()))
    devices = os.getenv("CUDA_VISIBLE_DEVICES", all_gpu_ids)

    torch.cuda.set_device(local_rank)

    _info(f"LOCAL_RANK: {local_rank} - CUDA_VISIBLE_DEVICES: [{devices}]")


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
    config.n_gpus = _GPUS.value

    # Setup logging
    logger_vis4d = logging.getLogger("vis4d")
    log_dir = os.path.join(config.output_dir, f"log_{config.timestamp}.txt")
    setup_logger(logger_vis4d, log_dir)

    rank_zero_info("Environment info: %s", get_pretty_env_info())

    if _SHOW_CONFIG.value:
        rank_zero_info("*" * 80)
        rank_zero_info(pprints_config(config))
        rank_zero_info("*" * 80)

    # Instantiate classes
    data_connector = instantiate_classes(config.data_connector)
    model = instantiate_classes(config.model)
    optimizers = instantiate_classes(config.optimizers)
    loss = instantiate_classes(config.loss)

    if "train_callbacks" in config and _MODE.value == "train":
        train_callbacks = instantiate_classes(config.train_callbacks)
    else:
        train_callbacks = None

    if "test_callbacks" in config:
        test_callbacks = instantiate_classes(config.test_callbacks)
    else:
        test_callbacks = None

    if config.n_gpus > 1:
        ddp_setup()

    train_dataloader = instantiate_classes(
        config.data.train_dataloader
    ).values()[0]

    test_dataloader = instantiate_classes(
        config.data.test_dataloader
    ).values()[0]

    if config.n_gpus == 0:
        device = torch.device("cpu")
    else:
        rank = get_local_rank()
        device = torch.device(f"cuda:{rank}")

    model.to(device)

    if config.n_gpus > 1:
        model = DDP(model, device_ids=[rank])

    tester = Tester(
        dataloaders=test_dataloader,
        data_connector=data_connector,
        test_callbacks=test_callbacks,
    )

    # TODO: Parameter sweep. Where to save the results? What name for the run?
    if _SWEEP.value is not None:
        # config = _CONFIG.value
        # sweep_obj = instantiate_classes(_SWEEP.value)

        # for config in replicate_config(
        #     _CONFIG.value,
        #     method=sweep_obj.method,
        #     sampling_args=sweep_obj.sampling_args,
        # ):
        #     train(config.value)
        pass
    elif _MODE.value == "train":
        trainer = Trainer(
            num_epochs=config.num_epochs,
            log_step=1,
            dataloaders=train_dataloader,
            data_connector=data_connector,
            train_callbacks=train_callbacks,
        )

        trainer.train(model, optimizers, loss, tester)
    elif _MODE.value == "test":
        tester.test(model)

    if config.n_gpus > 1:
        destroy_process_group()


if __name__ == "__main__":
    app.run(main)
