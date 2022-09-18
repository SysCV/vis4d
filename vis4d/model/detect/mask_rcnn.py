"""Mask RCNN model implementation and runtime."""
from typing import List, Optional, Union

import torch
from torch import nn

from vis4d.common_to_revise.datasets import bdd100k_track_map, coco_det_map
from vis4d.common_to_revise.optimizers import sgd, step_schedule
from vis4d.engine_to_revise import BaseCLI
from vis4d.op.base.resnet import ResNet
from vis4d.op.detect.data import InsSegDataModule
from vis4d.op.detect.faster_rcnn import (
    FasterRCNNHead,
    FRCNNOut,
    get_default_anchor_generator,
    get_default_rcnn_box_encoder,
    get_default_rpn_box_encoder,
)
from vis4d.op.detect.faster_rcnn_test import normalize
from vis4d.op.detect.rcnn import (
    Det2Mask,
    DetOut,
    MaskRCNNHead,
    MaskRCNNLoss,
    RCNNLoss,
    RoI2Det,
    postprocess_dets,
)
from vis4d.op.detect.rpn import RPNLoss
from vis4d.op.detect.util import apply_mask
from vis4d.op.fpp.fpn import FPN
from vis4d.op.utils import load_model_checkpoint
from vis4d.optim import DefaultOptimizer
from vis4d.struct_to_revise import (
    Boxes2D,
    InputSample,
    InstanceMasks,
    LossesType,
    ModelOutput,
)


class MaskRCNN(nn.Module):
    """Mask RCNN model."""

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
        self.mask_head = MaskRCNNHead()
        self.rpn_loss = RPNLoss(anchor_gen, rpn_bbox_encoder)
        self.rcnn_loss = RCNNLoss(rcnn_bbox_encoder)
        self.mask_rcnn_loss = MaskRCNNLoss()
        self.transform_outs = RoI2Det(rcnn_bbox_encoder)
        self.det2mask = Det2Mask()

        if weights == "mmdet":
            from vis4d.op.detect.mask_rcnn_test import MASK_REV_KEYS, REV_KEYS

            weights = (
                "mmdet://mask_rcnn/mask_rcnn_r50_fpn_2x_coco/"
                "mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_"
                "20200505_003907-3e542a40.pth"
            )
            load_model_checkpoint(self.backbone, weights, REV_KEYS)
            load_model_checkpoint(self.fpn, weights, REV_KEYS)
            load_model_checkpoint(self.faster_rcnn_heads, weights, REV_KEYS)
            load_model_checkpoint(self.mask_head, weights, MASK_REV_KEYS)
        elif weights is not None:
            load_model_checkpoint(self, weights)

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
        images, images_hw, target_boxes, target_classes, target_masks = (
            normalize(data.images.tensor),
            [(wh[1], wh[0]) for wh in data.images.image_sizes],
            [x.boxes for x in data.targets.boxes2d],
            [x.class_ids for x in data.targets.boxes2d],
            [x.masks for x in data.targets.instance_masks],
        )
        ######

        features = self.fpn(self.backbone(images))
        outputs = self.faster_rcnn_heads(
            features, images_hw, target_boxes, target_classes
        )
        mask_outs = self.mask_head(
            features[2:-1], outputs.sampled_proposals.boxes
        )

        rpn_losses = self.rpn_loss(*outputs.rpn, target_boxes, images_hw)
        rcnn_losses = self.rcnn_loss(
            *outputs.roi,
            outputs.sampled_proposals.boxes,
            outputs.sampled_targets.labels,
            outputs.sampled_targets.boxes,
            outputs.sampled_targets.classes,
        )
        assert outputs.sampled_target_indices is not None
        sampled_masks = apply_mask(
            outputs.sampled_target_indices, target_masks
        )[0]
        mask_losses = self.mask_rcnn_loss(
            mask_outs.mask_pred,
            outputs.sampled_proposals.boxes,
            outputs.sampled_targets.classes,
            sampled_masks,
        )

        ### boilerplate interfacing code
        losses = dict(
            **rpn_losses._asdict(),
            **rcnn_losses._asdict(),
            **mask_losses._asdict(),
        )
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
        orig_wh = [(original_wh[1], original_wh[0])]
        ######

        features = self.fpn(self.backbone(images))
        outs = self.faster_rcnn_heads(features, images_hw)
        boxes, scores, class_ids = self.transform_outs(
            *outs.roi, outs.proposals.boxes, images_hw
        )
        mask_outs = self.mask_head(features[2:-1], boxes)
        post_dets = DetOut(
            boxes=postprocess_dets(boxes, images_hw, orig_wh),
            scores=scores,
            class_ids=class_ids,
        )
        masks = self.det2mask(
            mask_outs=mask_outs.mask_pred.sigmoid(),
            dets=post_dets,
            images_hw=orig_wh,
        )

        ### boilerplate interfacing code
        dets = Boxes2D(
            torch.cat([boxes[0], scores[0].unsqueeze(-1)], -1),
            class_ids[0],
        )
        mask_pred = InstanceMasks(
            masks.masks[0], class_ids[0], score=scores[0], detections=dets
        )
        output = {
            "detect": [
                dets.to_scalabel({i: s for s, i in coco_det_map.items()})
            ],
            "ins_seg": [
                mask_pred.to_scalabel({i: s for s, i in coco_det_map.items()})
            ],
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
        num_classes = len(bdd100k_track_map)
    elif experiment == "coco":
        num_classes = len(coco_det_map)
    else:
        raise NotImplementedError(f"Experiment {experiment} not known!")

    model = MaskRCNN(num_classes=num_classes, weights=weights)
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

    python -m vis4d.model.detect.mask_rcnn fit --data.experiment coco --trainer.gpus 6,7 --data.samples_per_gpu 8 --data.workers_per_gpu 8"""
    DetectCLI(model_class=setup_model, datamodule_class=InsSegDataModule)
