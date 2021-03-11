"""Detection training API."""

import os
from typing import Dict, Optional

from detectron2.config import CfgNode
from detectron2.engine import DefaultTrainer, launch
from detectron2.evaluation import COCOEvaluator, DatasetEvaluator

from openmt.config import Config

from .config import default_setup, to_detectron2


class Trainer(DefaultTrainer):  # type: ignore
    """Trainer with COCOEvaluator for testing."""

    @classmethod
    def build_evaluator(
        cls, cfg: CfgNode, dataset_name: str
    ) -> DatasetEvaluator:
        """Build COCOEvaluator."""
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)


def train_func(
    det2cfg: CfgNode, cfg: Config
) -> Optional[Dict[str, Dict[str, float]]]:
    """Training function."""
    trainer = Trainer(det2cfg)
    trainer.resume_or_load(resume=cfg.launch.resume)
    return trainer.train()  # type: ignore


def train(cfg: Config) -> None:
    """Launcher for training."""
    detectron2cfg = to_detectron2(cfg)
    default_setup(detectron2cfg, cfg.launch)

    launch(
        train_func,
        cfg.launch.num_gpus,
        num_machines=cfg.launch.num_machines,
        machine_rank=cfg.launch.machine_rank,
        dist_url=cfg.launch.dist_url,
        args=(detectron2cfg, cfg),
    )
