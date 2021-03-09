"""Detection training API."""

import os
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator


class Trainer(DefaultTrainer):
    """Trainer with COCOEvaluator for testing."""
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)


def train(args, cfg):
    """Training function."""

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()
