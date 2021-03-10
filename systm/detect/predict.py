"""Detection prediction API."""

from argparse import Namespace
from typing import Dict

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode
from detectron2.engine import launch

from .train import Trainer


def predict_func(cfg: CfgNode, resume: bool) -> Dict[str, Dict[str, float]]:
    """Prediction function."""
    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=resume
    )

    return Trainer.test(cfg, model)  # type: ignore


def predict(args: Namespace, cfg: CfgNode) -> None:
    """Launcher for prediction."""
    launch(
        predict_func,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(cfg, args.resume),
    )
