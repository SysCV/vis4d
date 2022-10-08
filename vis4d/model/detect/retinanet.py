"""RetinaNet model implementation and runtime."""
from typing import Optional, Union

import torchvision
from torch import nn

from vis4d.common import COMMON_KEYS, DictData
from vis4d.common.detect_data import DetectDataModule
from vis4d.data.datasets.coco import coco_det_map
from vis4d.op.base.resnet import ResNet
from vis4d.op.box.util import bbox_postprocess
from vis4d.op.detect.retinanet import (
    Dense2Det,
    RetinaNetHead,
    RetinaNetLoss,
    get_default_box_matcher,
    get_default_box_sampler,
)
from vis4d.op.fpp.fpn import FPN, LastLevelP6P7
from vis4d.op.utils import load_model_checkpoint
from vis4d.optim import DefaultOptimizer
from vis4d.pl import BaseCLI
from vis4d.pl.defaults import sgd, step_schedule
from vis4d.run.data.datasets import bdd100k_det_map
from vis4d.struct_to_revise import LossesType, ModelOutput

REV_KEYS = [
    (r"^bbox_head\.", "retinanet_head."),
    (r"^backbone\.", "backbone.body."),
    (r"^neck.lateral_convs\.", "fpn.inner_blocks."),
    (r"^neck.fpn_convs\.", "fpn.layer_blocks."),
    (r"^fpn.layer_blocks.3\.", "fpn.extra_blocks.p6."),
    (r"^fpn.layer_blocks.4\.", "fpn.extra_blocks.p7."),
    (r"\.conv.weight", ".weight"),
    (r"\.conv.bias", ".bias"),
]


class RetinaNet(nn.Module):
    """RetinaNet wrapper class for checkpointing etc."""

    def __init__(
        self, num_classes: int, weights: Optional[str] = None
    ) -> None:
        """Init."""
        super().__init__()
        self.backbone = ResNet("resnet50", pretrained=True, trainable_layers=3)
        self.fpn = FPN(
            self.backbone.out_channels[3:],
            256,
            LastLevelP6P7(2048, 256),
            start_index=3,
        )
        self.retinanet_head = RetinaNetHead(
            num_classes=num_classes, in_channels=256
        )
        self.retinanet_loss = RetinaNetLoss(
            self.retinanet_head.anchor_generator,
            self.retinanet_head.box_encoder,
            get_default_box_matcher(),
            get_default_box_sampler(),
            torchvision.ops.sigmoid_focal_loss,
        )
        self.transform_outs = Dense2Det(
            self.retinanet_head.anchor_generator,
            self.retinanet_head.box_encoder,
            num_pre_nms=1000,
            max_per_img=100,
            nms_threshold=0.5,
            score_thr=0.05,
        )

        if weights == "mmdet":
            weights = (
                "mmdet://retinanet/retinanet_r50_fpn_2x_coco/"
                "retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth"
            )
            load_model_checkpoint(self, weights, rev_keys=REV_KEYS)
        elif weights is not None:
            load_model_checkpoint(self, weights)

    def forward(self, data: DictData) -> Union[LossesType, ModelOutput]:
        """Forward."""
        if self.training:
            return self._forward_train(data)
        return self._forward_test(data)

    def _forward_train(self, data: DictData) -> LossesType:
        """Forward training stage."""
        images, images_hw, target_boxes, target_classes = (
            data[COMMON_KEYS.images],
            data[COMMON_KEYS.metadata]["input_hw"],
            data[COMMON_KEYS.boxes2d],
            data[COMMON_KEYS.boxes2d_classes],
        )

        features = self.fpn(self.backbone(images))
        outputs = self.retinanet_head(features[-5:])
        losses = self.retinanet_loss(
            outputs.cls_score,
            outputs.bbox_pred,
            target_boxes,
            images_hw,
            target_classes,
        )
        return dict(**losses._asdict())

    def _forward_test(self, data: DictData) -> ModelOutput:
        """Forward testing stage."""
        images = data[COMMON_KEYS.images]
        original_hw = data[COMMON_KEYS.metadata]["original_hw"]
        images_hw = data[COMMON_KEYS.metadata]["input_hw"]

        features = self.fpn(self.backbone(images))
        outs = self.retinanet_head(features[-5:])
        boxes, scores, class_ids = self.transform_outs(
            class_outs=outs.cls_score,
            regression_outs=outs.bbox_pred,
            images_hw=images_hw,
        )
        for i, boxs in enumerate(boxes):
            boxes[i] = bbox_postprocess(boxs, original_hw[i], images_hw[i])
        output = dict(
            boxes2d=boxes, boxes2d_scores=scores, boxes2d_classes=class_ids
        )
        return output


def setup_model(
    experiment: str,
    lr: float = 0.01,
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

    model = RetinaNet(num_classes=num_classes, weights=weights)
    return DefaultOptimizer(
        model,
        optimizer_init=sgd(lr),
        lr_scheduler_init=step_schedule(max_epochs),
    )


class DetectCLI(BaseCLI):
    """Detect CLI."""

    def add_arguments_to_parser(self, parser):
        """Link data and model experiment argument."""
        parser.link_arguments("data.experiment", "model.experiment")
        parser.link_arguments("model.max_epochs", "trainer.max_epochs")


if __name__ == "__main__":
    """Example:

    python -m vis4d.model.detect.retinanet fit --data.experiment coco --trainer.gpus 6,7 --data.samples_per_gpu 8 --data.workers_per_gpu 8"""
    DetectCLI(model_class=setup_model, datamodule_class=DetectDataModule)
