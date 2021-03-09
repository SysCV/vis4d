"""Detection prediction API."""

from .train import  Trainer
from detectron2.checkpoint import DetectionCheckpointer


def predict(args, cfg):
    """Prediction function."""

    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    res = Trainer.test(cfg, model)
    return res
