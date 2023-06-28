"""Implementation of a single experiment.

Helper functions to execute a single experiment.

This will be called by the CLI for each experiment configuration.
"""
from __future__ import annotations

import logging
import os

import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.collect_env import get_pretty_env_info

from vis4d.common.distributed import (
    broadcast,
    get_local_rank,
    get_rank,
    get_world_size,
)
from vis4d.common.logging import (
    _info,
    rank_zero_info,
    rank_zero_warn,
    setup_logger,
)
from vis4d.common.slurm import init_dist_slurm
from vis4d.common.util import init_random_seed, set_random_seed, set_tf32
from vis4d.config import instantiate_classes
from vis4d.config.common.types import ExperimentConfig

from .optim import set_up_optimizers
from .parser import pprints_config
from .trainer import Trainer


def ddp_setup(
    torch_distributed_backend: str = "nccl", slurm: bool = False
) -> None:
    """Setup DDP environment and init processes.

    Args:
        torch_distributed_backend (str): Backend to use (`nccl` or `gloo`)
        slurm (bool): If set, setup slurm running jobs.
    """
    if slurm:
        init_dist_slurm()

    global_rank = get_rank()
    world_size = get_world_size()
    _info(
        f"Initializing distributed: "
        f"GLOBAL_RANK: {global_rank}, MEMBER: {global_rank + 1}/{world_size}"
    )
    init_process_group(
        torch_distributed_backend, rank=global_rank, world_size=world_size
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


def run_experiment(
    config: ExperimentConfig,
    mode: str,
    num_gpus: int = 0,
    show_config: bool = False,
    use_slurm: bool = False,
) -> None:
    """Entry point for running a single experiment.

    Args:
        config (ExperimentConfig): Configuration dictionary.
        mode (str): Mode to run the experiment in. Either `fit` or `test`.
        num_gpus (int): Number of GPUs to use.
        show_config (bool): If set, prints the configuration.
        use_slurm (bool): If set, setup slurm running jobs. This will set the
            required environment variables for slurm.

    Raises:
        ValueError: If `mode` is not `fit` or `test`.
    """
    # Setup logging
    logger_vis4d = logging.getLogger("vis4d")
    log_dir = os.path.join(config.output_dir, f"log_{config.timestamp}.txt")
    setup_logger(logger_vis4d, log_dir)

    rank_zero_info("Environment info: %s", get_pretty_env_info())

    # PyTorch Setting
    set_tf32(False)
    if "benchmark" in config:
        torch.backends.cudnn.benchmark = config.benchmark

    # TODO: Add random seed and DDP
    if show_config:
        rank_zero_info(pprints_config(config))

    # Instantiate classes
    model = instantiate_classes(config.model)

    if config.get("sync_batchnorm", False):
        if num_gpus > 1:
            rank_zero_info(
                "SyncBN enabled, converting BatchNorm layers to"
                " SyncBatchNorm layers."
            )
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        else:
            rank_zero_warn(
                "use_sync_bn is True, but not in a distributed setting."
                " BatchNorm layers are not converted."
            )

    # Callbacks
    callbacks = [instantiate_classes(cb) for cb in config.callbacks]

    # Setup DDP & seed
    seed = config.get("seed", init_random_seed())
    if num_gpus > 1:
        ddp_setup(slurm=use_slurm)

        # broadcast seed to all processes
        seed = broadcast(seed)

    # Setup Dataloaders & seed
    if mode == "fit":
        set_random_seed(seed)
        _info(f"[rank {get_rank()}] Global seed set to {seed}")
        train_dataloader = instantiate_classes(config.data.train_dataloader)
        train_data_connector = instantiate_classes(config.train_data_connector)
        optimizers = set_up_optimizers(config.optimizers, model)
        loss = instantiate_classes(config.loss)
    else:
        train_dataloader = None
        train_data_connector = None

    test_dataloader = instantiate_classes(config.data.test_dataloader)
    test_data_connector = instantiate_classes(config.test_data_connector)

    # Setup Model
    if num_gpus == 0:
        device = torch.device("cpu")
    else:
        rank = get_local_rank()
        device = torch.device(f"cuda:{rank}")

    model.to(device)

    if num_gpus > 1:
        model = DDP(  # pylint: disable=redefined-variable-type
            model, device_ids=[rank]
        )

    # Setup Callbacks
    for cb in callbacks:
        cb.setup()

    trainer = Trainer(
        device=device,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        train_data_connector=train_data_connector,
        test_data_connector=test_data_connector,
        callbacks=callbacks,
        num_epochs=config.params.get("num_epochs", -1),
        num_steps=config.params.get("num_steps", -1),
        check_val_every_n_epoch=config.get("check_val_every_n_epoch", 1),
        val_check_interval=config.get("val_check_interval", None),
        use_ema_model_for_test=config.get("use_ema_model_for_test", False),
    )

    if mode == "fit":
        trainer.fit(model, optimizers, loss)
    elif mode == "test":
        trainer.test(model)

    if num_gpus > 1:
        destroy_process_group()
