"""RetinaNet model implementation and runtime."""
from typing import List, Optional, Tuple, Union

import torch
import torchvision
from torch import nn

from vis4d.op.base.resnet import ResNet
from vis4d.op.box.matchers import BaseMatcher
from vis4d.op.box.samplers import BaseSampler
from vis4d.op.box.encoder import BoxEncoder2D
from vis4d.op.box.util import bbox_postprocess
from vis4d.op.detect.anchor_generator import AnchorGenerator
from vis4d.op.detect.retinanet import (
    Dense2Det,
    RetinaNetHead,
    RetinaNetHeadLoss,
    RetinaNetOut,
)
from vis4d.op.fpp.fpn import FPN, LastLevelP6P7
from vis4d.op.utils import load_model_checkpoint
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

    def forward(
        self,
        images: torch.Tensor,
        images_hw: Optional[List[Tuple[int, int]]] = None,
        original_hw: Optional[List[Tuple[int, int]]] = None,
    ) -> Union[RetinaNetOut, ModelOutput]:
        """Forward."""
        if self.training:
            return self.forward_train(images)
        assert images_hw is not None and original_hw is not None
        return self.forward_test(images, images_hw, original_hw)

    def forward_train(self, images: torch.Tensor) -> RetinaNetOut:
        """Forward training stage."""
        features = self.fpn(self.backbone(images))
        return self.retinanet_head(features[-5:])

    def forward_test(
        self,
        images: torch.Tensor,
        images_hw: List[Tuple[int, int]],
        original_hw: List[Tuple[int, int]],
    ) -> ModelOutput:
        """Forward testing stage."""
        features = self.fpn(self.backbone(images))
        outs = self.retinanet_head(features[-5:])
        boxes, scores, class_ids = self.transform_outs(
            class_outs=outs.cls_score,
            regression_outs=outs.bbox_pred,
            images_hw=images_hw,
        )
        for i, boxs in enumerate(boxes):
            boxes[i] = bbox_postprocess(boxs, original_hw[i], images_hw[i])
        return dict(
            boxes2d=boxes, boxes2d_scores=scores, boxes2d_classes=class_ids
        )


class RetinaNetLoss(nn.Module):
    """RetinaNet Loss."""

    def __init__(
        self,
        anchor_generator: AnchorGenerator,
        box_encoder: BoxEncoder2D,
        box_matcher: BaseMatcher,
        box_sampler: BaseSampler,
    ) -> None:
        """Init."""
        super().__init__()
        self.retinanet_loss = RetinaNetHeadLoss(
            anchor_generator,
            box_encoder,
            box_matcher,
            box_sampler,
            torchvision.ops.sigmoid_focal_loss,
        )

    def forward(
        self,
        outputs: RetinaNetOut,
        images_hw: List[Tuple[int, int]],
        target_boxes: List[torch.Tensor],
        target_classes: List[torch.Tensor],
    ) -> LossesType:
        """Forward."""
        losses = self.retinanet_loss(
            outputs.cls_score,
            outputs.bbox_pred,
            target_boxes,
            images_hw,
            target_classes,
        )
        return losses._asdict()