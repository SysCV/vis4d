"""One-stage detector runtime configuration."""
from vis4d.common.datasets import bdd100k_det_map, coco_det_map
from vis4d.common.models import build_retinanet, build_yolox
from vis4d.common.optimizers import sgd, step_schedule
from vis4d.detect.data import DetectDataModule
from vis4d.detect.two_stage import DetectCLI
from vis4d.op.optimize import DefaultOptimizer


def setup_model(
    experiment: str,
    lr: float = 0.02,
    max_epochs: int = 12,
    detector: str = "RetinaNet",
) -> DefaultOptimizer:
    """Setup model with experiment specific hyperparameters."""
    if experiment == "bdd100k":
        category_mapping = bdd100k_det_map
    elif experiment == "coco":
        category_mapping = coco_det_map
    else:
        raise NotImplementedError(f"Experiment {experiment} not known!")

    if detector == "RetinaNet":
        model = build_retinanet(category_mapping)
    elif detector == "YOLOX":
        model = build_yolox(category_mapping)
    else:
        raise NotImplementedError(f"Detector {detector} not known!")

    return DefaultOptimizer(
        model,
        lr_scheduler_init=step_schedule(max_epochs),
        optimizer_init=sgd(lr),
    )


if __name__ == "__main__":
    DetectCLI(
        model_class=setup_model,
        datamodule_class=DetectDataModule,
    )
