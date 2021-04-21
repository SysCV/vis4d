"""Tracking prediction API."""
from typing import Dict

import torch
from bdd100k.eval.mot import EvalResults
from detectron2.config import CfgNode
from detectron2.engine import launch

from openmt.config import Config
from openmt.detect.config import default_setup, to_detectron2
from openmt.model import build_model

from .checkpointer import TrackingCheckpointer
from .train import TrackingTrainer


def track_predict_func(
    det2cfg: CfgNode, cfg: Config
) -> Dict[str, EvalResults]:
    """Prediction function."""
    model = build_model(cfg.model)
    model.to(torch.device(cfg.launch.device))
    if hasattr(model, "detector") and hasattr(model.detector, "d2_cfg"):
        det2cfg.MODEL.merge_from_other_cfg(model.detector.d2_cfg.MODEL)
    if cfg.launch.weights != "detectron2":
        det2cfg.MODEL.WEIGHTS = cfg.launch.weights
    TrackingCheckpointer(model, save_dir=det2cfg.OUTPUT_DIR).resume_or_load(
        det2cfg.MODEL.WEIGHTS, resume=cfg.launch.resume
    )
    result = TrackingTrainer.test_static(cfg, det2cfg, model)
    return result


def predict(cfg: Config) -> None:
    """Launcher for prediction."""
    detectron2cfg = to_detectron2(cfg)
    default_setup(detectron2cfg, cfg.launch)

    launch(
        track_predict_func,
        cfg.launch.num_gpus,
        num_machines=cfg.launch.num_machines,
        machine_rank=cfg.launch.machine_rank,
        dist_url=cfg.launch.dist_url,
        args=(detectron2cfg, cfg),
    )
