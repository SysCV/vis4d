"""RetinaNet model implementation and runtime."""

from __future__ import annotations

from torch import Tensor, nn

from vis4d.common import LossesType
from vis4d.common.ckpt import load_model_checkpoint
from vis4d.op.base.resnet import ResNet
from vis4d.op.box.anchor import AnchorGenerator
from vis4d.op.box.box2d import scale_and_clip_boxes
from vis4d.op.box.encoder import DeltaXYWHBBoxEncoder
from vis4d.op.box.matchers import Matcher
from vis4d.op.box.samplers import Sampler
from vis4d.op.detect.common import DetOut
from vis4d.op.detect.retinanet import (
    Dense2Det,
    RetinaNetHead,
    RetinaNetHeadLoss,
    RetinaNetOut,
)
from vis4d.op.fpp.fpn import FPN, ExtraFPNBlock

REV_KEYS = [
    (r"^backbone\.", "basemodel."),
    (r"^bbox_head\.", "retinanet_head."),
    (r"^neck.lateral_convs\.", "fpn.inner_blocks."),
    (r"^neck.fpn_convs\.", "fpn.layer_blocks."),
    (r"^fpn.layer_blocks.3\.", "fpn.extra_blocks.convs.0."),
    (r"^fpn.layer_blocks.4\.", "fpn.extra_blocks.convs.1."),
    (r"\.conv.weight", ".weight"),
    (r"\.conv.bias", ".bias"),
]


class RetinaNet(nn.Module):
    """RetinaNet wrapper class for checkpointing etc."""

    def __init__(self, num_classes: int, weights: None | str = None) -> None:
        """Creates an instance of the class.

        Args:
            num_classes (int): Number of classes.
            weights (None | str, optional): Weights to load for model. If
                set to "mmdet", will load MMDetection pre-trained weights.
                Defaults to None.
        """
        super().__init__()
        self.basemodel = ResNet(
            "resnet50", pretrained=True, trainable_layers=3
        )
        self.fpn = FPN(
            self.basemodel.out_channels[3:],
            256,
            ExtraFPNBlock(2, 2048, 256, add_extra_convs="on_input"),
            start_index=3,
        )
        self.retinanet_head = RetinaNetHead(
            num_classes=num_classes, in_channels=256
        )
        self.transform_outs = Dense2Det(
            self.retinanet_head.anchor_generator,
            self.retinanet_head.box_decoder,
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
        images: Tensor,
        input_hw: None | list[tuple[int, int]] = None,
        original_hw: None | list[tuple[int, int]] = None,
    ) -> RetinaNetOut | DetOut:
        """Forward pass.

        Args:
            images (Tensor): Input images.
            input_hw (None | list[tuple[int, int]], optional): Input image
                resolutions. Defaults to None.
            original_hw (None | list[tuple[int, int]], optional): Original
                image resolutions (before padding and resizing). Required for
                testing. Defaults to None.

        Returns:
            RetinaNetOut | DetOut: Either raw model outputs (for training) or
                predicted outputs (for testing).
        """
        if self.training:
            return self.forward_train(images)
        assert input_hw is not None and original_hw is not None
        return self.forward_test(images, input_hw, original_hw)

    def forward_train(self, images: Tensor) -> RetinaNetOut:
        """Forward training stage.

        Args:
            images (Tensor): Input images.

        Returns:
            RetinaNetOut: Raw model outputs.
        """
        features = self.fpn(self.basemodel(images))
        return self.retinanet_head(features[-5:])

    def forward_test(
        self,
        images: Tensor,
        images_hw: list[tuple[int, int]],
        original_hw: list[tuple[int, int]],
    ) -> DetOut:
        """Forward testing stage.

        Args:
            images (Tensor): Input images.
            images_hw (list[tuple[int, int]]): Input image resolutions.
            original_hw (list[tuple[int, int]]): Original image resolutions
                (before padding and resizing).

        Returns:
            DetOut: Predicted outputs.
        """
        features = self.fpn(self.basemodel(images))
        outs = self.retinanet_head(features[-5:])
        boxes, scores, class_ids = self.transform_outs(
            cls_outs=outs.cls_score,
            reg_outs=outs.bbox_pred,
            images_hw=images_hw,
        )
        for i, boxs in enumerate(boxes):
            boxes[i] = scale_and_clip_boxes(boxs, original_hw[i], images_hw[i])
        return DetOut(boxes, scores, class_ids)


class RetinaNetLoss(nn.Module):
    """RetinaNet Loss."""

    def __init__(
        self,
        anchor_generator: AnchorGenerator,
        box_encoder: DeltaXYWHBBoxEncoder,
        box_matcher: Matcher,
        box_sampler: Sampler,
    ) -> None:
        """Creates an instance of the class.

        Args:
            anchor_generator (AnchorGenerator): Anchor generator for RPN.
            box_encoder (BoxEncoder2D): Bounding box encoder.
            box_matcher (BaseMatcher): Bounding box matcher.
            box_sampler (BaseSampler): Bounding box sampler.
        """
        super().__init__()
        self.retinanet_loss = RetinaNetHeadLoss(
            anchor_generator, box_encoder, box_matcher, box_sampler
        )

    def forward(
        self,
        outputs: RetinaNetOut,
        images_hw: list[tuple[int, int]],
        target_boxes: list[Tensor],
        target_classes: list[Tensor],
    ) -> LossesType:
        """Forward of loss function.

        Args:
            outputs (RetinaNetOut): Raw model outputs.
            images_hw (list[tuple[int, int]]): Input image resolutions.
            target_boxes (list[Tensor]): Bounding box labels.
            target_classes (list[Tensor]): Class labels.

        Returns:
            LossesType: Dictionary of model losses.
        """
        losses = self.retinanet_loss(
            outputs.cls_score,
            outputs.bbox_pred,
            target_boxes,
            images_hw,
            target_classes,
        )
        return losses._asdict()
