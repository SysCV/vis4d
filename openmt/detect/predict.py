"""Detection prediction API."""
from typing import Dict

import torch
from detectron2.checkpoint import DetectionCheckpointer

from openmt.config import Config
from openmt.model import build_model

from .config import default_setup, to_detectron2
from .train import Trainer


def predict(cfg: Config) -> Dict[str, Dict[str, float]]:
    """Prediction function."""
    det2cfg = to_detectron2(cfg)
    default_setup(det2cfg, cfg.launch)
    model = build_model(cfg.model)
    model.to(torch.device(cfg.launch.device))
    if hasattr(model, "detector") and hasattr(model.detector, "d2_cfg"):
        det2cfg.MODEL.merge_from_other_cfg(model.detector.d2_cfg.MODEL)
    if cfg.launch.weights != "detectron2":
        det2cfg.MODEL.WEIGHTS = cfg.launch.weights
    DetectionCheckpointer(model, save_dir=det2cfg.OUTPUT_DIR).resume_or_load(
        det2cfg.MODEL.WEIGHTS, resume=cfg.launch.resume
    )
    return Trainer.test(det2cfg, model)  # type: ignore
