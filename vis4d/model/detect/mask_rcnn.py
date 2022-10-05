"""Mask RCNN model implementation and runtime."""
from typing import Optional, Union

from torch import nn

from vis4d.common_to_revise.datasets import bdd100k_track_map
from vis4d.common_to_revise.detect_data import InsSegDataModule
from vis4d.common_to_revise.optimizers import sgd, step_schedule
from vis4d.data.datasets.base import DataKeys, DictData
from vis4d.data.datasets.coco import coco_det_map
from vis4d.op.base.resnet import ResNet
from vis4d.op.box.util import apply_mask, bbox_postprocess
from vis4d.op.detect.faster_rcnn import (
    FasterRCNNHead,
    get_default_anchor_generator,
    get_default_rcnn_box_encoder,
    get_default_rpn_box_encoder,
)
from vis4d.op.detect.rcnn import (
    Det2Mask,
    DetOut,
    MaskRCNNHead,
    MaskRCNNLoss,
    RCNNLoss,
    RoI2Det,
)
from vis4d.op.detect.rpn import RPNLoss
from vis4d.op.fpp.fpn import FPN
from vis4d.op.utils import load_model_checkpoint
from vis4d.optim import DefaultOptimizer
from vis4d.pl import CLI
from vis4d.struct_to_revise import LossesType, ModelOutput

REV_KEYS = [
    (r"^rpn_head.rpn_reg\.", "rpn_head.rpn_box."),
    (r"^roi_head.bbox_head\.", "roi_head."),
    (r"^roi_head.mask_head\.", "mask_head."),
    (r"^convs\.", "mask_head.convs."),
    (r"^upsample\.", "mask_head.upsample."),
    (r"^conv_logits\.", "mask_head.conv_logits."),
    (r"^roi_head\.", "faster_rcnn_heads.roi_head."),
    (r"^rpn_head\.", "faster_rcnn_heads.rpn_head."),
    (r"^backbone\.", "backbone.body."),
    (r"^neck.lateral_convs\.", "fpn.inner_blocks."),
    (r"^neck.fpn_convs\.", "fpn.layer_blocks."),
    (r"\.conv.weight", ".weight"),
    (r"\.conv.bias", ".bias"),
]


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
            weights = (
                "mmdet://mask_rcnn/mask_rcnn_r50_fpn_2x_coco/"
                "mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_"
                "20200505_003907-3e542a40.pth"
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
        device = next(self.parameters()).device  # TODO hack for now
        images, images_hw, target_boxes, target_classes, target_masks = (
            data[DataKeys.images].to(device),
            data[DataKeys.metadata]["input_hw"],
            [b.to(device) for b in data[DataKeys.boxes2d]],
            [b.to(device) for b in data[DataKeys.boxes2d_classes]],
            [m.to(device) for m in data[DataKeys.masks]],
        )

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

        return dict(
            **rpn_losses._asdict(),
            **rcnn_losses._asdict(),
            **mask_losses._asdict(),
        )

    def _forward_test(self, data: DictData) -> ModelOutput:
        """Forward testing stage."""
        device = next(self.parameters()).device  # TODO hack for now
        images = data[DataKeys.images].to(device)
        original_hw = data[DataKeys.metadata]["original_hw"]
        images_hw = data[DataKeys.metadata]["input_hw"]

        features = self.fpn(self.backbone(images))
        outs = self.faster_rcnn_heads(features, images_hw)
        boxes, scores, class_ids = self.transform_outs(
            *outs.roi, outs.proposals.boxes, images_hw
        )
        mask_outs = self.mask_head(features[2:-1], boxes)
        for i, boxs in enumerate(boxes):
            boxes[i] = bbox_postprocess(boxs, original_hw[i], images_hw[i])
        post_dets = DetOut(boxes=boxes, scores=scores, class_ids=class_ids)
        masks = self.det2mask(
            mask_outs=mask_outs.mask_pred.sigmoid(),
            dets=post_dets,
            images_hw=original_hw,
        )

        output = dict(
            boxes2d=boxes,
            boxes2d_scores=scores,
            boxes2d_classes=class_ids,
            masks=masks.masks,
        )
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


class DetectCLI(CLI):
    """Detect CLI."""

    def add_arguments_to_parser(self, parser):
        """Link data and model experiment argument."""
        parser.link_arguments("data.experiment", "model.experiment")
        parser.link_arguments("model.max_epochs", "trainer.max_epochs")


if __name__ == "__main__":
    """Example:

    python -m vis4d.model.detect.mask_rcnn fit --data.experiment coco --trainer.gpus 6,7 --data.samples_per_gpu 8 --data.workers_per_gpu 8"""
    DetectCLI(model_class=setup_model, datamodule_class=InsSegDataModule)
