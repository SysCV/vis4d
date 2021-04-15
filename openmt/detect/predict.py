"""Detection prediction API."""
from typing import Dict

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode
from detectron2.engine import launch

from openmt.config import Config
from openmt.model import build_model

from .config import default_setup, to_detectron2
from .train import Trainer


def predict_func(det2cfg: CfgNode, cfg: Config) -> Dict[str, Dict[str, float]]:
    """Prediction function."""
    model = build_model(cfg.model)
    model.to(torch.device(cfg.launch.device))
    if hasattr(model, "detector") and hasattr(model.detector, "d2_cfg"):
        det2cfg.MODEL.merge_from_other_cfg(model.detector.d2_cfg.MODEL)
    DetectionCheckpointer(model, save_dir=det2cfg.OUTPUT_DIR).resume_or_load(
        det2cfg.MODEL.WEIGHTS, resume=cfg.launch.resume
    )

    return Trainer.test(det2cfg, model)  # type: ignore


def predict(cfg: Config) -> None:
    """Launcher for prediction."""
    detectron2cfg = to_detectron2(cfg)
    default_setup(detectron2cfg, cfg.launch)

    launch(
        predict_func,
        cfg.launch.num_gpus,
        num_machines=cfg.launch.num_machines,
        machine_rank=cfg.launch.machine_rank,
        dist_url=cfg.launch.dist_url,
        args=(detectron2cfg, cfg),
    )
