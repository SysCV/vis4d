"""Config preparation functions for the detection module."""
import os
from typing import List

import torch
from detectron2 import model_zoo
from detectron2.config import CfgNode, get_cfg
from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils import comm
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.env import seed_all_rng
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger

from openmt.config import Config, Dataset, DatasetType, Launch

model_mapping = {
    "faster-rcnn": "COCO-Detection/faster_rcnn_",
    "retinanet": "COCO-Detection/retinanet_",
    "mask-rcnn": "COCO-InstanceSegmentation/mask_rcnn_",
}

backbone_mapping = {
    "r101-fpn": "R_101_FPN_3x.yaml",
    "r101-c4": "R_101_C4_3x.yaml",
    "r101-dc5": "R_101_DC5_3x.yaml",
    "r50-fpn": "R_50_FPN_3x.yaml",
    "r50-c4": "R_50_C4_3x.yaml",
    "r50-dc5": "R_50_DC5_3x.yaml",
}


def _register(datasets: List[Dataset]) -> List[str]:
    """Register datasets in detectron2."""
    names = []
    for dataset in datasets:
        if not dataset.type == DatasetType.COCO:
            raise NotImplementedError(
                "Currently only COCO style dataset structure is supported."
            )
        try:
            DatasetCatalog.get(dataset.name)
        except KeyError:
            register_coco_instances(
                dataset.name, {}, dataset.annotation_file, dataset.data_root
            )
            names.append(dataset.name)
            continue
        print(
            "WARNING: You tried to register the same dataset name "
            f"twice. Skipping instance:\n{str(dataset)}"
        )
    return names


def to_detectron2(config: Config) -> CfgNode:
    """Convert a Config object to a detectron2 readable configuration."""
    cfg = get_cfg()

    # load model base config, checkpoint
    detectron2_model_string = None
    if os.path.exists(config.detection.model_base):
        base_cfg = config.detection.model_base
    else:
        if config.detection.override_mapping:
            detectron2_model_string = config.detection.model_base
        else:
            model, backbone = config.detection.model_base.split("/")
            detectron2_model_string = (
                model_mapping[model] + backbone_mapping[backbone]
            )
        base_cfg = model_zoo.get_config_file(detectron2_model_string)

    cfg.merge_from_file(base_cfg)

    # load checkpoint
    if config.detection.weights is not None:
        if os.path.exists(config.detection.weights):
            cfg.MODEL.WEIGHTS = config.detection.weights
        elif (
            config.detection.weights == "detectron2"
            and detectron2_model_string is not None
        ):
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                detectron2_model_string
            )
        else:
            raise ValueError(
                f"model weights path {config.detection.weights} "
                f"not "
                f"found. If you're loading a detectron2 config from local, "
                f"please also specify a local checkpoint file"
            )

    # convert model attributes
    if config.detection.num_classes:
        if config.detection.device is not None:
            cfg.MODEL.DEVICE = config.detection.device
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = config.detection.num_classes
        cfg.MODEL.RETINANET.NUM_CLASSES = config.detection.num_classes
    cfg.OUTPUT_DIR = config.output_dir

    # convert solver attributes
    if config.solver is not None:
        cfg.SOLVER.IMS_PER_BATCH = config.solver.images_per_batch
        cfg.SOLVER.LR_SCHEDULER_NAME = config.solver.lr_policy
        cfg.SOLVER.BASE_LR = config.solver.base_lr
        cfg.SOLVER.MAX_ITER = config.solver.max_iters
        if config.solver.checkpoint_period is not None:
            cfg.SOLVER.STEPS = config.solver.steps
        if config.solver.checkpoint_period is not None:
            cfg.SOLVER.CHECKPOINT_PERIOD = config.solver.checkpoint_period
        if config.solver.checkpoint_period is not None:
            cfg.TEST.EVAL_PERIOD = config.solver.eval_period

    # convert dataloader attributes
    if config.dataloader is not None:
        cfg.DATALOADER.NUM_WORKERS = config.dataloader.num_workers

    # register datasets
    if config.train is not None:
        cfg.DATASETS.TRAIN = _register(config.train)
    if config.test is not None:
        cfg.DATASETS.TEST = _register(config.test)

    cfg.freeze()
    return cfg


def default_setup(cfg: CfgNode, args: Launch) -> None:
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

    logger.info("Launch configuration: %s", str(args))

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
    if not args.eval_only:
        torch.backends.cudnn.benchmark = cfg.CUDNN_BENCHMARK
