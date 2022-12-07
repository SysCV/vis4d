"""Default run configuration for pytorch lightning."""
from typing import Optional

from vis4d.data.datasets.bdd100k import bdd100k_det_map
from vis4d.data.datasets.coco import coco_det_map
from vis4d.engine.model.faster_rcnn import data_connector
from vis4d.model.detect.faster_rcnn import FasterRCNN, FasterRCNNLoss
from vis4d.pl.data import DetectDataModule
from vis4d.pl.defaults import sgd, step_schedule

from ..optimizer import DefaultOptimizer
from ..trainer import CLI


def setup_model(  # pylint: disable=invalid-name
    experiment: str,
    lr: float = 0.02,
    max_epochs: int = 12,
    weights: Optional[str] = None,
) -> DefaultOptimizer:
    """Setup model with experiment specific hyperparameters."""
    if experiment == "bdd100k":
        num_classes = len(bdd100k_det_map)
    elif experiment == "coco":
        num_classes = len(coco_det_map)
    else:
        raise NotImplementedError(f"Experiment {experiment} not known!")

    model = FasterRCNN(num_classes=num_classes, weights=weights)
    loss = FasterRCNNLoss()

    return DefaultOptimizer(
        model,
        loss,
        data_connector=data_connector,
        optimizer_init=sgd(lr),
        lr_scheduler_init=step_schedule(max_epochs),
    )


class DefaultCLI(CLI):
    """Default CLI for running models with pytorch lightning."""

    def add_arguments_to_parser(self, parser):
        """Link data and model experiment argument."""
        # parser.link_arguments("data.experiment", "model.experiment") TODO
        parser.link_arguments("model.max_epochs", "trainer.max_epochs")


if __name__ == "__main__":
    """Example:

    >>> python -m vis4d.pl.model.faster_rcnn fit --data.experiment coco --trainer.gpus 6,7 --data.samples_per_gpu 8 --data.workers_per_gpu 8"""
    DefaultCLI(model_class=setup_model, datamodule_class=DetectDataModule)
