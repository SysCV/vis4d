"""One-stage detector runtime configuration."""
from projects.common.datasets import bdd100k_det_map, coco_det_map
from projects.common.models import build_retinanet, build_yolox
from projects.common.optimizers import sgd, step_schedule
from projects.detect.data import DetectDataModule
from projects.detect.two_stage import DetectCLI
from vis4d.model.detect.mmdet import MMOneStageDetector


def setup_model(
    experiment: str,
    lr: float = 0.02,
    max_epochs: int = 12,
    detector: str = "FRCNN",
) -> MMOneStageDetector:
    """Setup model with experiment specific hyperparameters."""
    if experiment == "bdd100k":
        category_mapping = bdd100k_det_map
    elif experiment == "coco":
        category_mapping = coco_det_map
    else:
        raise NotImplementedError(f"Experiment {experiment} not known!")

    model_kwargs = {
        "lr_scheduler_init": step_schedule(max_epochs),
        "optimizer_init": sgd(lr),
    }

    if detector == "RetinaNet":
        model = build_retinanet(category_mapping, model_kwargs=model_kwargs)
    elif detector == "YOLOX":
        model = build_yolox(category_mapping, model_kwargs=model_kwargs)
    else:
        raise NotImplementedError(f"Detector {detector} not known!")

    return model


if __name__ == "__main__":
    DetectCLI(
        model_class=setup_model,
        datamodule_class=DetectDataModule,
    )
