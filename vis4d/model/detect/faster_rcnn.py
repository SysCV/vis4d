"""Faster RCNN model implementation and runtime."""
from typing import List, Optional, Union

import torch
from torch import nn

from vis4d.common_to_revise.datasets import bdd100k_det_map, coco_det_map
from vis4d.common_to_revise.optimizers import sgd, step_schedule
from vis4d.engine_to_revise import BaseCLI
from vis4d.op.base.resnet import ResNet
from vis4d.op.detect.data import DetectDataModule
from vis4d.op.detect.faster_rcnn import (
    FasterRCNNHead,
    FRCNNOut,
    get_default_anchor_generator,
    get_default_rcnn_box_encoder,
    get_default_rpn_box_encoder,
)
from vis4d.op.detect.faster_rcnn_test import normalize
from vis4d.op.detect.rcnn import RCNNLoss, RoI2Det
from vis4d.op.detect.rpn import RPNLoss
from vis4d.op.fpp.fpn import FPN
from vis4d.op.utils import load_model_checkpoint
from vis4d.optim import DefaultOptimizer
from vis4d.struct_to_revise import (
    Boxes2D,
    InputSample,
    LossesType,
    ModelOutput,
)


class FasterRCNN(nn.Module):
    """Faster RCNN model."""

    def __init__(
        self, num_classes: int, weights: Optional[str] = None
    ) -> None:
        """Init."""
        super().__init__()
        anchor_gen = get_default_anchor_generator()
        rpn_bbox_encoder = get_default_rpn_box_encoder()
        rcnn_bbox_encoder = get_default_rcnn_box_encoder()
        self.backbone = ResNet("resnet50", pretrained=True, trainable_layers=3)
        self.fpn = FPN(self.backbone.out_channels[2:], 256)
        self.faster_rcnn_heads = FasterRCNNHead(
            num_classes=num_classes,
            anchor_generator=anchor_gen,
            rpn_box_encoder=rpn_bbox_encoder,
            rcnn_box_encoder=rcnn_bbox_encoder,
        )
        self.rpn_loss = RPNLoss(anchor_gen, rpn_bbox_encoder)
        self.rcnn_loss = RCNNLoss(rcnn_bbox_encoder)
        self.transform_outs = RoI2Det(rcnn_bbox_encoder)

        if weights == "mmdet":
            from vis4d.op.detect.faster_rcnn_test import REV_KEYS

            weights = "mmdet://faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
            load_model_checkpoint(self.backbone, weights, REV_KEYS)
            load_model_checkpoint(self.fpn, weights, REV_KEYS)
            load_model_checkpoint(self.faster_rcnn_heads, weights, REV_KEYS)
        elif weights is not None:
            load_model_checkpoint(self, weights)

    def visualize_proposals(
        self, images: torch.Tensor, outs: FRCNNOut, topk: int = 100
    ) -> None:
        """Visualize topk proposals."""
        from vis4d.vis.image import imshow_bboxes

        for im, boxes, scores in zip(images, *outs.proposals):
            _, topk_indices = torch.topk(scores, topk)
            imshow_bboxes(im, boxes[topk_indices])

    def forward(
        self, data: List[InputSample]
    ) -> Union[LossesType, ModelOutput]:
        """Forward."""
        if self.training:
            return self._forward_train(data)
        return self._forward_test(data)

    def _forward_train(self, data: List[InputSample]) -> LossesType:
        """Forward training stage."""
        ### boilerplate interfacing code
        data = data[0]
        images, images_hw, target_boxes, target_classes = (
            normalize(data.images.tensor),
            [(wh[1], wh[0]) for wh in data.images.image_sizes],
            [x.boxes for x in data.targets.boxes2d],
            [x.class_ids for x in data.targets.boxes2d],
        )
        ######

        features = self.fpn(self.backbone(images))
        outputs = self.faster_rcnn_heads(
            features, images_hw, target_boxes, target_classes
        )

        rpn_losses = self.rpn_loss(*outputs.rpn, target_boxes, images_hw)
        rcnn_losses = self.rcnn_loss(
            *outputs.roi,
            outputs.sampled_proposals.boxes,
            outputs.sampled_targets.labels,
            outputs.sampled_targets.boxes,
            outputs.sampled_targets.classes,
        )

        ### boilerplate interfacing code
        losses = dict(**rpn_losses._asdict(), **rcnn_losses._asdict())
        ######
        return losses

    def _forward_test(self, data: List[InputSample]) -> ModelOutput:
        """Forward testing stage."""
        ### boilerplate interfacing code
        data = data[0]
        images = normalize(data.images.tensor)
        original_wh = (
            data.metadata[0].size.width,
            data.metadata[0].size.height,
        )
        output_wh = data.images.image_sizes[0]
        images_hw = [(output_wh[1], output_wh[0])]
        ######

        features = self.fpn(self.backbone(images))
        outs = self.faster_rcnn_heads(features, images_hw)
        boxes, scores, class_ids = self.transform_outs(
            *outs.roi, outs.proposals.boxes, images_hw
        )

        ### boilerplate interfacing code
        dets = Boxes2D(
            torch.cat([boxes[0], scores[0].unsqueeze(-1)], -1),
            class_ids[0],
        )
        dets.postprocess(original_wh, output_wh)
        output = {
            "detect": [
                dets.to_scalabel({i: s for s, i in coco_det_map.items()})
            ]
        }
        ######
        return output


def setup_model(
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

    python -m vis4d.model.detect.faster_rcnn fit --data.experiment coco --trainer.gpus 6,7 --data.samples_per_gpu 8 --data.workers_per_gpu 8"""
    DetectCLI(
        model_class=setup_model,
        datamodule_class=DetectDataModule,
    )
