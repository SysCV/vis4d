"""Config preparation functions for the detection module."""
import itertools
import logging
import os
from typing import Dict, List

import torch
from detectron2.config import CfgNode, get_cfg
from detectron2.data import DatasetCatalog
from detectron2.utils import comm
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.env import seed_all_rng
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger
from devtools import debug
from scalabel.label.typing import Frame

from openmt.config import Config, Launch
from openmt.data.datasets import register_dataset
from openmt.data.datasets.base import BaseDatasetConfig


def gather_predictions(
    predictions: Dict[str, List[Frame]]
) -> Dict[str, List[Frame]]:
    """Gather prediction dict in distributed setting."""
    comm.synchronize()
    predictions_list = comm.gather(predictions, dst=0)

    result = {}
    for key in predictions:
        prediction_list = [p[key] for p in predictions_list]
        result[key] = list(itertools.chain(*prediction_list))
    return result


def _register(datasets: List[BaseDatasetConfig]) -> List[str]:
    """Register datasets in detectron2."""
    names = []
    for dataset in datasets:
        names.append(dataset.name)
        try:
            DatasetCatalog.get(dataset.name)
        except KeyError:
            register_dataset(dataset)
            continue
        logger = logging.getLogger(__name__)
        logger.info(
            "WARNING: You tried to register the same dataset name "
            "twice. Skipping instance:\n%s",
            dataset,
        )
    return names


def register_directory(input_path: str) -> str:
    """Register directory containing input data as dataset."""
    if input_path[-1] == "/":
        input_path = input_path[:-1]
    dataset_name = os.path.basename(input_path)
    dataset = BaseDatasetConfig(
        type="Custom", name=dataset_name, data_root=input_path
    )
    register_dataset(dataset)
    return dataset_name


def to_detectron2(config: Config) -> CfgNode:
    """Convert a Config object to a detectron2 readable configuration."""
    cfg = get_cfg()
    cfg.OUTPUT_DIR = config.launch.output_dir

    # convert solver attributes
    if config.solver is not None:
        cfg.SOLVER.IMS_PER_BATCH = config.solver.images_per_gpu
        if config.launch.num_gpus > 0:
            cfg.SOLVER.IMS_PER_BATCH *= config.launch.num_gpus
        cfg.SOLVER.LR_SCHEDULER_NAME = config.solver.lr_policy
        cfg.SOLVER.BASE_LR = config.solver.base_lr
        cfg.SOLVER.MAX_ITER = config.solver.max_iters
        if config.solver.warmup_iters is not None:
            cfg.SOLVER.WARMUP_ITERS = config.solver.warmup_iters
        if config.solver.steps is not None:
            cfg.SOLVER.STEPS = config.solver.steps
        if config.solver.checkpoint_period is not None:
            cfg.SOLVER.CHECKPOINT_PERIOD = config.solver.checkpoint_period
        if config.solver.eval_period is not None:
            cfg.TEST.EVAL_PERIOD = config.solver.eval_period

    # convert dataloader attributes
    if config.dataloader is not None:
        cfg.DATALOADER.NUM_WORKERS = config.dataloader.workers_per_gpu

    # register datasets
    if config.train is not None:
        cfg.DATASETS.TRAIN = _register(config.train)
    if config.test is not None:
        cfg.DATASETS.TEST = _register(config.test)

    return cfg


def default_setup(cfg: Config, det2cfg: CfgNode, args: Launch) -> None:
    """Perform some basic common setups at the beginning of a job.

    1. Set up the logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory
    """
    rank = comm.get_rank()
    logger = setup_logger(
        det2cfg.OUTPUT_DIR, distributed_rank=rank, name="openmt"
    )
    setup_logger(
        os.path.join(det2cfg.OUTPUT_DIR, "d2_log.txt"),
        distributed_rank=rank,
        name="detectron2",
    )

    logger.info(
        "Rank of current process: %s. World size: %s",
        rank,
        comm.get_world_size(),
    )
    logger.info("Environment info: %s", collect_env_info())

    logger.info(
        "Running with full config:\n %s",
        str(debug.format(cfg)).split("\n", 1)[1],
    )
    if comm.is_main_process():
        # Note: some of the detectron2 scripts may expect the existence of
        # config.yaml in output directory
        path = os.path.join(det2cfg.OUTPUT_DIR, "config.yaml")
        with PathManager.open(path, "w") as f:
            f.write(det2cfg.dump())
        # save openMT config
        path = os.path.join(det2cfg.OUTPUT_DIR, "config.json")
        with PathManager.open(path, "w") as f:
            f.write(cfg.json())
        logger.info("Full config saved to %s", path)

    # make sure each worker has different, yet deterministic seed if specified
    if cfg.launch.seed < 0:
        seed_all_rng()
    else:
        logger.info("Using a fixed random seed: %s", cfg.launch.seed)
        seed_all_rng(cfg.launch.seed + rank)

    # cudnn benchmark has large overhead. It shouldn't be used considering the
    # small size of typical validation set.
    if args.action == "train":
        torch.backends.cudnn.benchmark = det2cfg.CUDNN_BENCHMARK
