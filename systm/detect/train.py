"""Detection training API."""

import os
from argparse import Namespace
from collections import OrderedDict
from typing import Optional

from detectron2.config import CfgNode
from detectron2.engine import DefaultTrainer, launch
from detectron2.evaluation import COCOEvaluator, DatasetEvaluator


class Trainer(DefaultTrainer):
    """Trainer with COCOEvaluator for testing."""

    @classmethod
    def build_evaluator(
        cls, cfg: CfgNode, dataset_name: str
    ) -> DatasetEvaluator:
        """Build COCOEvaluator."""
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)


def train_func(cfg: CfgNode, resume: bool) -> Optional[OrderedDict]:
    """Training function."""
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=resume)
    return trainer.train()


def train(args: Namespace, cfg: CfgNode) -> None:
    """Launcher for training."""
    launch(
        train_func,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(cfg, args.resume),
    )
