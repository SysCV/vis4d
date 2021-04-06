"""Tracking prediction API."""

from typing import Dict

from detectron2.config import CfgNode
from detectron2.engine import launch

from openmt.config import Config
from openmt.detect.config import default_setup, to_detectron2
from openmt.modeling.meta_arch import build_model

from .checkpointer import TrackingCheckpointer
from .train import TrackingTrainer


def track_predict_func(
    det2cfg: CfgNode, cfg: Config
) -> Dict[str, Dict[str, float]]:
    """Prediction function."""
    model = build_model(cfg)
    TrackingCheckpointer(model, save_dir=det2cfg.OUTPUT_DIR).resume_or_load(
        det2cfg.MODEL.WEIGHTS, resume=cfg.launch.resume
    )

    return TrackingTrainer.test(det2cfg, model)  # type: ignore


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
