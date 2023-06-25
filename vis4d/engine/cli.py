"""CLI interface."""
from __future__ import annotations

import logging
import os

import torch
from absl import app, flags
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.collect_env import get_pretty_env_info

from vis4d.common import ArgsType
from vis4d.common.ckpt import load_model_checkpoint
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

from .optim import set_up_optimizers
from .parser import DEFINE_config_file, pprints_config
from .trainer import Trainer

# TODO: Currently this does not allow to load multpile config files.
# Would be nice to extend functionality to chain multiple config files using
# e.g. --config=model_1.py --config=loader_args.py
# or --config=my_config.py --config.train_dl=different_dl.py

# TODO: Support resume from folder and load config directly from it.
_CONFIG = DEFINE_config_file("config", method_name="get_config")
_SWEEP = DEFINE_config_file("sweep", method_name="get_sweep")
_GPUS = flags.DEFINE_integer("gpus", default=0, help="Number of GPUs")
_CKPT = flags.DEFINE_string("ckpt", default=None, help="Checkpoint path")
_RESUME = flags.DEFINE_bool("resume", default=False, help="Resume training")
_SHOW_CONFIG = flags.DEFINE_bool(
    "print-config", default=False, help="If set, prints the configuration."
)
_SLURM = flags.DEFINE_bool(
    "slurm", default=False, help="If set, setup slurm running jobs."
)


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


def main(argv: ArgsType) -> None:
    """Main entry point for the CLI.

    Example to run this script:
    >>> python -m vis4d.engine.cli --config configs/faster_rcnn/faster_rcnn_coco.py
    """
    # Get config
    assert len(argv) > 1, "Mode must be specified: `fit` or `test`"
    mode = argv[1]
    assert mode in {"fit", "test"}, f"Invalid mode: {mode}"
    config = _CONFIG.value
    num_gpus = _GPUS.value

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
    if _SHOW_CONFIG.value:
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
        ddp_setup(slurm=_SLURM.value)

        # broadcast seed to all processes
        seed = broadcast(seed)

    # Setup Dataloaders & seed
    if mode == "fit":
        set_random_seed(seed)
        _info(f"[rank {get_rank()}] Global seed set to {seed}")
        train_dataloader = instantiate_classes(config.data.train_dataloader)
        train_data_connector = instantiate_classes(config.train_data_connector)
        optimizers = set_up_optimizers(config.optimizers, [model])
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

    # Checkpoint path
    ckpt_path = _CKPT.value

    # Resume training
    resume = _RESUME.value
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
        global_step = ckpt["global_step"] + 1

        for i, optim in enumerate(optimizers):
            optim.optimizer.load_state_dict(ckpt["optimizers"][i])

            if ckpt["lr_schedulers"][i] is not None:
                assert optim.lr_scheduler is not None
                optim.lr_scheduler.load_state_dict(ckpt["lr_schedulers"][i])
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
    )

    if resume:
        rank_zero_info(
            f"Restored all states from the checkpoint at {ckpt_path}"
        )

    # TODO: Parameter sweep. Where to save the results? What name for the run?
    if _SWEEP.value is not None:
        # config = _CONFIG.value
        # sweep_obj = instantiate_classes(_SWEEP.value)

        # from vis4d.config.replicator import replicate_config

        # for config in replicate_config(
        #     _CONFIG.value,
        #     method=sweep_obj.method,
        #     sampling_args=sweep_obj.sampling_args,
        # ):
        #     train(config.value)
        pass
    elif mode == "fit":
        trainer.fit(model, optimizers, loss)
    elif mode == "test":
        trainer.test(model)

    if num_gpus > 1:
        destroy_process_group()


if __name__ == "__main__":
    app.run(main)
