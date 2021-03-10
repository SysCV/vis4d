"""Default configs and boilerplate logic for default behavior of systm."""

import argparse
import os
import sys

import torch
from detectron2.config import CfgNode
from detectron2.utils import comm
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.env import seed_all_rng
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger

__all__ = [
    "default_argument_parser",
    "default_setup",
]


def default_argument_parser():
    """Create a parser with common systm arguments."""
    parser = argparse.ArgumentParser(description="systm options")
    parser.add_argument(
        "action",
        type=str,
        choices=["train", "predict"],
        help="Action to execute",
    )
    parser.add_argument(
        "--config", default="", metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory.",
    )
    parser.add_argument(
        "--eval-only", action="store_true", help="perform evaluation only"
    )
    parser.add_argument(
        "--num-gpus", type=int, default=1, help="number of gpus *per machine*"
    )
    parser.add_argument(
        "--num-machines", type=int, default=1, help="total number of machines"
    )
    parser.add_argument(
        "--machine-rank",
        type=int,
        default=0,
        help="the rank of this machine (unique per machine)",
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = (
        2 ** 15
        + 2 ** 14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs at the end of "
        "the command. "
        "See config references at https://detectron2.readthedocs.io/"
        "modules/config.html#config-references",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def default_setup(cfg: CfgNode, args: argparse.Namespace):
    """Perform some basic common setups at the beginning of a job.

    1. Set up the logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory
    """
    rank = comm.get_rank()
    setup_logger(cfg.OUTPUT_DIR, distributed_rank=rank, name="fvcore")
    logger = setup_logger(cfg.OUTPUT_DIR, distributed_rank=rank)

    logger.info(
        "Rank of current process: %s. World size: %s",
        rank,
        comm.get_world_size(),
    )
    logger.info("Environment info: %s", collect_env_info())

    logger.info("Command line arguments: %s", str(args))
    if hasattr(args, "config_file") and args.config_file != "":
        logger.info(
            "Contents of args.config_file=%s:\n%s",
            args.config_file,
            PathManager.open(args.config_file, "r").read(),
        )

    logger.info("Running with full config:\n %s", cfg)
    if comm.is_main_process():
        # Note: some of the detectron2 scripts may expect the existence of
        # config.yaml in output directory
        path = os.path.join(cfg.OUTPUT_DIR, "config.yaml")
        with PathManager.open(path, "w") as f:
            f.write(cfg.dump())
        logger.info("Full config saved to %s", path)

    # make sure each worker has different, yet deterministic seed if specified
    seed_all_rng(None if cfg.SEED < 0 else cfg.SEED + rank)

    # cudnn benchmark has large overhead. It shouldn't be used considering the
    # small size of typical validation set.
    if not (hasattr(args, "eval_only") and args.eval_only):
        torch.backends.cudnn.benchmark = cfg.CUDNN_BENCHMARK
