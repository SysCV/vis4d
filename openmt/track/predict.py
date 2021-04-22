"""Tracking prediction API."""
from typing import Dict

import torch
from bdd100k.eval.mot import EvalResults

from openmt.config import Config
from openmt.detect.config import default_setup, to_detectron2
from openmt.model import build_model

from .checkpointer import TrackingCheckpointer
from .train import TrackingTrainer


def predict(cfg: Config) -> Dict[str, EvalResults]:
    """Prediction function."""
    det2cfg = to_detectron2(cfg)
    default_setup(det2cfg, cfg.launch)

    model = build_model(cfg.model)
    model.to(torch.device(cfg.launch.device))
    if hasattr(model, "detector") and hasattr(model.detector, "d2_cfg"):
        det2cfg.MODEL.merge_from_other_cfg(model.detector.d2_cfg.MODEL)
    if cfg.launch.weights != "detectron2":
        det2cfg.MODEL.WEIGHTS = cfg.launch.weights  # pragma: no cover
    TrackingCheckpointer(model, save_dir=det2cfg.OUTPUT_DIR).resume_or_load(
        det2cfg.MODEL.WEIGHTS, resume=cfg.launch.resume
    )
    result = TrackingTrainer.test_static(cfg, det2cfg, model)
    return result
