"""YOLOX model implementation and runtime."""

from __future__ import annotations

import torch
from torch import nn

from vis4d.common.ckpt import load_model_checkpoint
from vis4d.op.base import BaseModel, CSPDarknet
from vis4d.op.box.box2d import scale_and_clip_boxes
from vis4d.op.detect.common import DetOut
from vis4d.op.detect.yolox import YOLOXHead, YOLOXOut, YOLOXPostprocess
from vis4d.op.fpp import YOLOXPAFPN, FeaturePyramidProcessing

REV_KEYS = [
    (r"^backbone\.", "basemodel."),
    (r"^bbox_head\.", "yolox_head."),
    (r"^neck\.", "fpn."),
    (r"\.bn\.", ".norm."),
    (r"\.conv.weight", ".weight"),
    (r"\.conv.bias", ".bias"),
]


class YOLOX(nn.Module):
    """YOLOX detector."""

    def __init__(
        self,
        num_classes: int,
        basemodel: BaseModel | None = None,
        fpn: FeaturePyramidProcessing | None = None,
        yolox_head: YOLOXHead | None = None,
        postprocessor: YOLOXPostprocess | None = None,
        weights: None | str = None,
    ) -> None:
        """Creates an instance of the class.

        Args:
            num_classes (int): Number of classes.
            basemodel (BaseModel, optional): Base model. Defaults to None. If
                None, will use CSPDarknet.
            fpn (FeaturePyramidProcessing, optional): Feature Pyramid
                Processing. Defaults to None. If None, will use YOLOXPAFPN.
            yolox_head (YOLOXHead, optional): YOLOX head. Defaults to None. If
                None, will use YOLOXHead.
            postprocessor (YOLOXPostprocess, optional): Post processor.
                Defaults to None. If None, will use YOLOXPostprocess.
            weights (None | str, optional): Weights to load for model. If
                set to "mmdet", will load MMDetection pre-trained weights.
                Defaults to None.
        """
        super().__init__()
        self.basemodel = (
            CSPDarknet(deepen_factor=0.33, widen_factor=0.5)
            if basemodel is None
            else basemodel
        )
        self.fpn = (
            YOLOXPAFPN([128, 256, 512], 128, num_csp_blocks=1)
            if fpn is None
            else fpn
        )
        self.yolox_head = (
            YOLOXHead(
                num_classes=num_classes, in_channels=128, feat_channels=128
            )
            if yolox_head is None
            else yolox_head
        )
        self.postprocessor = (
            YOLOXPostprocess(
                self.yolox_head.point_generator,
                self.yolox_head.box_decoder,
                nms_threshold=0.65,
                score_thr=0.01,
            )
            if postprocessor is None
            else postprocessor
        )

        if weights is not None:
            if weights.startswith("mmdet://") or weights.startswith(
                "bdd100k://"
            ):
                load_model_checkpoint(self, weights, rev_keys=REV_KEYS)
            else:
                load_model_checkpoint(self, weights)

    def forward(
        self,
        images: torch.Tensor,
        input_hw: None | list[tuple[int, int]] = None,
        original_hw: None | list[tuple[int, int]] = None,
    ) -> YOLOXOut | DetOut:
        """Forward pass.

        Args:
            images (torch.Tensor): Input images.
            input_hw (None | list[tuple[int, int]], optional): Input image
                resolutions. Defaults to None.
            original_hw (None | list[tuple[int, int]], optional): Original
                image resolutions (before padding and resizing). Required for
                testing. Defaults to None.

        Returns:
            YOLOXOut | DetOut: Either raw model outputs (for training) or
                predicted outputs (for testing).
        """
        if self.training:
            return self.forward_train(images)
        assert input_hw is not None and original_hw is not None
        return self.forward_test(images, input_hw, original_hw)

    def forward_train(self, images: torch.Tensor) -> YOLOXOut:
        """Forward training stage.

        Args:
            images (torch.Tensor): Input images.

        Returns:
            YOLOXOut: Raw model outputs.
        """
        features = self.fpn(self.basemodel(images.contiguous()))
        return self.yolox_head(features[-3:])

    def forward_test(
        self,
        images: torch.Tensor,
        images_hw: list[tuple[int, int]],
        original_hw: list[tuple[int, int]],
    ) -> DetOut:
        """Forward testing stage.

        Args:
            images (torch.Tensor): Input images.
            images_hw (list[tuple[int, int]]): Input image resolutions.
            original_hw (list[tuple[int, int]]): Original image resolutions
                (before padding and resizing).

        Returns:
            DetOut: Predicted outputs.
        """
        features = self.fpn(self.basemodel(images))
        outs = self.yolox_head(features[-3:])
        boxes, scores, class_ids = self.postprocessor(
            cls_outs=outs.cls_score,
            reg_outs=outs.bbox_pred,
            obj_outs=outs.objectness,
            images_hw=images_hw,
        )
        for i, boxs in enumerate(boxes):
            boxes[i] = scale_and_clip_boxes(boxs, original_hw[i], images_hw[i])
        return DetOut(boxes, scores, class_ids)
