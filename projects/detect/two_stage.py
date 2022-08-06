"""Two-stage detector runtime configuration."""
from projects.common.datasets import bdd100k_det_map, coco_det_map
from projects.common.models import build_faster_rcnn
from projects.common.optimizers import sgd, step_schedule
from projects.detect.data import DetectDataModule
from vis4d.engine.trainer import BaseCLI
from vis4d.model.optimize import DefaultOptimizer


def setup_model(
    experiment: str,
    lr: float = 0.02,
    max_epochs: int = 12,
    detector: str = "FRCNN",
) -> DefaultOptimizer:
    """Setup model with experiment specific hyperparameters."""
    if experiment == "bdd100k":
        category_mapping = bdd100k_det_map
    elif experiment == "coco":
        category_mapping = coco_det_map
    else:
        raise NotImplementedError(f"Experiment {experiment} not known!")

    if detector == "FRCNN":
        model = build_faster_rcnn(category_mapping)
    else:
        raise NotImplementedError(f"Detector {detector} not known!")

    return DefaultOptimizer(
        model,
        lr_scheduler_init=step_schedule(max_epochs),
        optimizer_init=sgd(lr),
    )


class DetectCLI(BaseCLI):
    """Detect CLI."""

    def add_arguments_to_parser(self, parser):
        """Link data and model experiment argument."""
        parser.link_arguments("data.experiment", "model.experiment")
        parser.link_arguments("model.max_epochs", "trainer.max_epochs")


if __name__ == "__main__":
    DetectCLI(
        model_class=setup_model,
        datamodule_class=DetectDataModule,
    )
