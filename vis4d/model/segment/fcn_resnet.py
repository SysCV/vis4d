"""FCN tests."""
from typing import Optional, Union, Tuple

import torch
from torch import nn

from vis4d.common_to_revise.datasets import bdd100k_seg_map
from vis4d.common_to_revise.optimizers import sgd, step_schedule
from vis4d.optim import DefaultOptimizer
from vis4d.op.base.resnet import ResNet
from vis4d.op.segment.fcn import FCNHead, FCNLoss, FCNOut


class FCN_ResNet(nn.Module):
    def __init__(self, base_model: str = "res", num_classes: int = 21) -> None:
        """Init."""
        super().__init__()
        if base_model.startswith("resnet"):
            self.basemodel = ResNet(
                base_model,
                pretrained=True,
                replace_stride_with_dilation=[False, True, True],
            )
        else:
            raise ValueError("base model not supported!")
        self.fcn = FCNHead(
            self.basemodel.out_channels[4:], num_classes, resize=(512, 512)
        )
        self.loss = FCNLoss([4, 5], nn.CrossEntropyLoss(ignore_index=255))

    def forward(
        self, images: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Union[Tuple[FCNOut, FCNLoss], FCNOut]:
        features = self.basemodel(images)
        out = self.fcn(features)
        if targets is not None:
            losses = self.loss(out.outputs, targets)
            return out, losses
        return out


def setup_model(
    experiment: str, lr: float = 0.02, max_epochs: int = 12
) -> DefaultOptimizer:
    """Setup model with experiment specific hyperparameters."""
    if experiment == "bdd100k":
        num_classes = len(bdd100k_seg_map)
    elif experiment == "coco":
        pass
        # num_classes = len(coco_seg_map)
    else:
        raise NotImplementedError(f"Experiment {experiment} not known!")

    model = FCN_ResNet(num_classes=num_classes)
    return DefaultOptimizer(
        model,
        optimizer_init=sgd(lr),
        lr_scheduler_init=step_schedule(max_epochs),
    )


class SegmentCLI(BaseCLI):
    """Segment CLI."""

    def add_arguments_to_parser(self, parser):
        """Link data and model experiment argument."""
        parser.link_arguments("data.experiment", "model.experiment")
        parser.link_arguments("model.max_steps", "trainer.max_steps")
        parser.link_arguments("trainer.gpus", "model.gpus")


if __name__ == "__main__":
    SegmentCLI(
        model_class=setup_model,
        datamodule_class=SegmentDataModule,
    )
