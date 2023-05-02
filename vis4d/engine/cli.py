"""CLI interface for vis4d.

Example to run this script:
>>> python -m vis4d.engine.cli --config vis4d/config/example/faster_rcnn_coco.py
"""
from __future__ import annotations

import logging
import os

import torch
from absl import app, flags
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.collect_env import get_pretty_env_info

from vis4d.common import ArgsType
from vis4d.common.distributed import (
    broadcast,
    get_local_rank,
    get_rank,
    get_world_size,
)
from vis4d.common.logging import _info, rank_zero_info, setup_logger
from vis4d.common.slurm import init_dist_slurm
from vis4d.common.util import init_random_seed, set_random_seed, set_tf32
from vis4d.config.parser import DEFINE_config_file
from vis4d.config.util import instantiate_classes, pprints_config
from vis4d.engine.optim import set_up_optimizers
from vis4d.engine.trainer import Trainer

# TODO: Currently this does not allow to load multpile config files.
# Would be nice to extend functionality to chain multiple config files using
# e.g. --config=model_1.py --config=loader_args.py
# or --config=my_config.py --config.train_dl=different_dl.py

_CONFIG = DEFINE_config_file("config", method_name="get_config")
_SWEEP = DEFINE_config_file("sweep", method_name="get_sweep")
_GPUS = flags.DEFINE_integer("gpus", default=0, help="Number of GPUs")
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
    data_connector = instantiate_classes(config.data_connector)
    model = instantiate_classes(config.model)
    optimizers = set_up_optimizers(config.optimizers, model)
    loss = instantiate_classes(config.loss)

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
    else:
        train_dataloader = None

    test_dataloader = instantiate_classes(config.data.test_dataloader)

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
        num_epochs=config.params.num_epochs,
        data_connector=data_connector,
        callbacks=callbacks,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
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
