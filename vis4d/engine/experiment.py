"""Implementation of a single experiment.

Helper functions to execute a single experiment.

This will be called for each experiment configuration.
"""

from __future__ import annotations

import logging
import os

import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.collect_env import get_pretty_env_info

from vis4d.common.ckpt import load_model_checkpoint
from vis4d.common.distributed import (
    broadcast,
    get_local_rank,
    get_rank,
    get_world_size,
)
from vis4d.common.logging import (
    _info,
    dump_config,
    rank_zero_info,
    rank_zero_warn,
    setup_logger,
)
from vis4d.common.slurm import init_dist_slurm
from vis4d.common.util import init_random_seed, set_random_seed, set_tf32
from vis4d.config import instantiate_classes
from vis4d.config.typing import ExperimentConfig

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
    ckpt_path: str | None = None,
    resume: bool = False,
) -> None:
    """Entry point for running a single experiment.

    Args:
        config (ExperimentConfig): Configuration dictionary.
        mode (str): Mode to run the experiment in. Either `fit` or `test`.
        num_gpus (int): Number of GPUs to use.
        show_config (bool): If set, prints the configuration.
        use_slurm (bool): If set, setup slurm running jobs. This will set the
            required environment variables for slurm.
        ckpt_path (str | None): Path to a checkpoint to load.
        resume (bool): If set, resume training from the checkpoint.

    Raises:
        ValueError: If `mode` is not `fit` or `test`.
    """
    # Setup logging
    logger_vis4d = logging.getLogger("vis4d")
    log_dir = os.path.join(config.output_dir, f"log_{config.timestamp}.txt")
    setup_logger(logger_vis4d, log_dir)

    # Dump config
    config_file = os.path.join(
        config.output_dir, f"config_{config.timestamp}.yaml"
    )
    dump_config(config, config_file)

    rank_zero_info("Environment info: %s", get_pretty_env_info())

    # PyTorch Setting
    set_tf32(config.use_tf32, config.tf32_matmul_precision)
    torch.hub.set_dir(f"{config.work_dir}/.cache/torch/hub")
    torch.backends.cudnn.benchmark = config.benchmark

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
    seed = init_random_seed() if config.seed == -1 else config.seed

    if num_gpus > 1:
        ddp_setup(slurm=use_slurm)

        # broadcast seed to all processes
        seed = broadcast(seed)

    # Setup Dataloaders & seed
    if mode == "fit":
        set_random_seed(seed)
        _info(f"[rank {get_rank()}] Global seed set to {seed}")
        train_dataloader = instantiate_classes(
            config.data.train_dataloader, seed=seed
        )
        train_data_connector = instantiate_classes(config.train_data_connector)
        optimizers, lr_schedulers = set_up_optimizers(
            config.optimizers, [model], len(train_dataloader)
        )
        loss = instantiate_classes(config.loss)
    else:
        train_dataloader = None
        train_data_connector = None

    if config.data.test_dataloader is not None:
        test_dataloader = instantiate_classes(config.data.test_dataloader)
    else:
        test_dataloader = None

    if config.test_data_connector is not None:
        test_data_connector = instantiate_classes(config.test_data_connector)
    else:
        test_data_connector = None

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

    # Resume training
    if resume:
        if ckpt_path is None:
            ckpt_path = os.path.join(
                config.output_dir, "checkpoints/last.ckpt"
            )
        rank_zero_info(
            f"Restoring states from the checkpoint path at {ckpt_path}"
        )
        ckpt = torch.load(ckpt_path, map_location="cpu")

        epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]

        for i, optimizer in enumerate(optimizers):
            optimizer.load_state_dict(ckpt["optimizers"][i])

        for i, lr_scheduler in enumerate(lr_schedulers):
            lr_scheduler.load_state_dict(ckpt["lr_schedulers"][i])
    else:
        epoch = 0
        global_step = 0

    if ckpt_path is not None:
        load_model_checkpoint(
            model,
            ckpt_path,
            rev_keys=[(r"^model\.", ""), (r"^module\.", "")],
        )

    trainer = Trainer(
        device=device,
        output_dir=config.output_dir,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        train_data_connector=train_data_connector,
        test_data_connector=test_data_connector,
        callbacks=callbacks,
        num_epochs=config.params.get("num_epochs", -1),
        num_steps=config.params.get("num_steps", -1),
        epoch=epoch,
        global_step=global_step,
        check_val_every_n_epoch=config.get("check_val_every_n_epoch", 1),
        val_check_interval=config.get("val_check_interval", None),
        log_every_n_steps=config.get("log_every_n_steps", 50),
    )

    if resume:
        rank_zero_info(
            f"Restored all states from the checkpoint at {ckpt_path}"
        )

    if mode == "fit":
        trainer.fit(model, optimizers, lr_schedulers, loss)
    elif mode == "test":
        trainer.test(model)

    if num_gpus > 1:
        destroy_process_group()
