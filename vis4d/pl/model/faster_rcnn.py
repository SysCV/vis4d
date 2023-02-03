# pylint: disable=consider-using-alias,consider-alternative-union-syntax
"""Default run configuration for pytorch lightning."""
from typing import Optional

import torch
from torch import nn

from vis4d.common.typing import LossesType
from vis4d.data.datasets.bdd100k import bdd100k_det_map
from vis4d.data.datasets.coco import coco_det_map
from vis4d.model.detect.faster_rcnn import FasterRCNN
from vis4d.op.detect.faster_rcnn import (
    FRCNNOut,
    get_default_anchor_generator,
    get_default_rcnn_box_encoder,
    get_default_rpn_box_encoder,
)
from vis4d.op.detect.rcnn import RCNNLoss
from vis4d.op.detect.rpn import RPNLoss
from vis4d.pl.data import DetectDataModule
from vis4d.pl.defaults import sgd, step_schedule
from vis4d.pl.optimizer import DefaultOptimizer
from vis4d.pl.trainer import CLI


# TODO, load from config
class FasterRCNNLoss(nn.Module):
    """Faster RCNN Loss."""

    def __init__(self) -> None:
        """Creates an instance of the class."""
        super().__init__()
        anchor_generator = get_default_anchor_generator()
        rpn_box_encoder = get_default_rpn_box_encoder()
        rcnn_box_encoder = get_default_rcnn_box_encoder()
        self.rpn_loss = RPNLoss(anchor_generator, rpn_box_encoder)
        self.rcnn_loss = RCNNLoss(rcnn_box_encoder)

    def forward(
        self,
        outputs: FRCNNOut,
        input_hw: list[tuple[int, int]],
        boxes2d: list[torch.Tensor],
    ) -> LossesType:
        """Forward of loss function.

        Args:
            outputs (FRCNNOut): Raw model outputs.
            input_hw (list[tuple[int, int]]): Input image resolutions.
            boxes2d (list[torch.Tensor]): Bounding box labels.

        Returns:
            LossesType: Dictionary of model losses.
        """
        rpn_losses = self.rpn_loss(*outputs.rpn, boxes2d, input_hw)
        assert (
            outputs.sampled_proposals is not None
            and outputs.sampled_targets is not None
        )
        rcnn_losses = self.rcnn_loss(
            *outputs.roi,
            outputs.sampled_proposals.boxes,
            outputs.sampled_targets.labels,
            outputs.sampled_targets.boxes,
            outputs.sampled_targets.classes,
        )
        return {**rpn_losses._asdict(), **rcnn_losses._asdict()}

    def __call__(
        self,
        outputs: FRCNNOut,
        input_hw: list[tuple[int, int]],
        boxes2d: list[torch.Tensor],
    ) -> LossesType:
        """Type definition for call implementation."""
        return self._call_impl(
            outputs,
            input_hw,
            boxes2d,
        )


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
        optimizer_init=sgd(lr),
        lr_scheduler_init=step_schedule(max_epochs),
    )


class DefaultCLI(CLI):
    """Default CLI for running models with pytorch lightning."""

    def add_arguments_to_parser(self, parser) -> None:
        """Link data and model experiment argument."""
        # parser.link_arguments("data.experiment", "model.experiment") TODO
        parser.link_arguments("model.max_epochs", "trainer.max_epochs")


if __name__ == "__main__":
    # pylint: disable=pointless-string-statement
    """Main function.

    Example Usage:
    >>> python -m vis4d.pl.model.faster_rcnn fit --data.experiment coco --trainer.gpus 6,7 --data.samples_per_gpu 8 --data.workers_per_gpu 8
    """
    DefaultCLI(model_class=setup_model, datamodule_class=DetectDataModule)
