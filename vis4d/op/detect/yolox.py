"""YOLOX detection head."""
from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import torch
from torch import nn
from torchvision.ops import batched_nms

from vis4d.op.box.anchor import MlvlPointGenerator
from vis4d.op.box.encoder import YOLOXBBoxDecoder
from vis4d.op.box.matchers import Matcher
from vis4d.op.box.samplers import Sampler
from vis4d.op.layer import Conv2d

from .common import DetOut


class YOLOXOut(NamedTuple):
    """YOLOX head outputs."""

    # Logits for box classification for each feature level. The logit
    # dimention is [batch_size, number of classes, height, width].
    cls_score: list[torch.Tensor]
    # Each box has regression for all classes for each feature level. So the
    # tensor dimension is [batch_size, 4, height, width].
    bbox_pred: list[torch.Tensor]
    objectness: list[torch.Tensor]  # TODO: Update docstring.


def get_default_point_generator() -> MlvlPointGenerator:
    """Get default point generator."""
    return MlvlPointGenerator(strides=[8, 16, 32], offset=0)


class YOLOXHead(nn.Module):
    """YOLOX Head.

    Args:
        num_classes (int): Number of classes.
        in_channels (int): Number of input channels.
        feat_channels (int, optional): Number of feature channels. Defaults to
            256.
        stacked_convs (int, optional): Number of stacked convolutions. Defaults
            to 2.
        strides (Sequence[int], optional): Strides for each feature level.
            Defaults to (8, 16, 32).
        point_generator (MlvlPointGenerator, optional): Point generator.
            Defaults to None.
        box_decoder (YOLOXBBoxDecoder, optional): Bounding box decoder.
            Defaults to None.
        box_matcher (Matcher, optional): Bounding box matcher. Defaults to
            None.
        box_sampler (Sampler, optional): Bounding box sampler. Defaults to
            None.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        feat_channels: int = 256,
        stacked_convs: int = 2,
        strides: Sequence[int] = (8, 16, 32),
        point_generator: MlvlPointGenerator | None = None,
        box_decoder: YOLOXBBoxDecoder | None = None,
        box_matcher: Matcher | None = None,
        box_sampler: Sampler | None = None,
    ):
        """Creates an instance of the class."""
        super().__init__()
        self.point_generator = (
            point_generator
            if point_generator is not None
            else get_default_point_generator()
        )
        if box_decoder is None:
            self.box_decoder = YOLOXBBoxDecoder()
        else:
            self.box_decoder = box_decoder
        self.box_matcher = box_matcher
        self.box_sampler = box_sampler

        self.multi_level_cls_convs = nn.ModuleList()
        self.multi_level_reg_convs = nn.ModuleList()
        self.multi_level_conv_cls = nn.ModuleList()
        self.multi_level_conv_reg = nn.ModuleList()
        self.multi_level_conv_obj = nn.ModuleList()
        for _ in strides:
            self.multi_level_cls_convs.append(
                self._build_stacked_convs(
                    in_channels, feat_channels, stacked_convs
                )
            )
            self.multi_level_reg_convs.append(
                self._build_stacked_convs(
                    in_channels, feat_channels, stacked_convs
                )
            )
            conv_cls, conv_reg, conv_obj = self._build_predictor(
                feat_channels, num_classes
            )
            self.multi_level_conv_cls.append(conv_cls)
            self.multi_level_conv_reg.append(conv_reg)
            self.multi_level_conv_obj.append(conv_obj)

    def _build_stacked_convs(
        self, in_channels: int, feat_channels: int, stacked_convs: int
    ) -> nn.Module:
        """Initialize conv layers of a single level head.

        Args:
            in_channels (int): Number of input channels.
            feat_channels (int): Number of feature channels.
            stacked_convs (int): Number of stacked conv layers.
        """
        stacked_conv_layers = []
        for i in range(stacked_convs):
            chn = in_channels if i == 0 else feat_channels
            stacked_conv_layers.append(
                Conv2d(
                    chn,
                    feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    norm=nn.BatchNorm2d(
                        feat_channels, eps=0.001, momentum=0.03
                    ),
                    activation=nn.SiLU(inplace=True),
                    bias=False,
                )
            )
        return nn.Sequential(*stacked_conv_layers)

    def _build_predictor(
        self, feat_channels: int, num_classes: int
    ) -> tuple[nn.Module, nn.Module, nn.Module]:
        """Initialize predictor layers of a single level head.

        Args:
            feat_channels (int): Number of input channels.
            num_classes (int): Number of classes.
        """
        conv_cls = nn.Conv2d(feat_channels, num_classes, 1)
        conv_reg = nn.Conv2d(feat_channels, 4, 1)
        conv_obj = nn.Conv2d(feat_channels, 1, 1)
        return conv_cls, conv_reg, conv_obj

    def forward(self, features: list[torch.Tensor]) -> YOLOXOut:
        """Forward pass of YOLOX head.

        Args:
            features (list[torch.Tensor]): Input features.

        Returns:
            YOLOXOut: Classification, box, and objectness predictions.
        """
        cls_score, bbox_pred, objectness = [], [], []
        for feature, cls_conv, reg_conv, conv_cls, conv_reg, conv_obj in zip(
            features,
            self.multi_level_cls_convs,
            self.multi_level_reg_convs,
            self.multi_level_conv_cls,
            self.multi_level_conv_reg,
            self.multi_level_conv_obj,
        ):
            cls_feat = cls_conv(feature)
            reg_feat = reg_conv(feature)

            cls_score.append(conv_cls(cls_feat))
            bbox_pred.append(conv_reg(reg_feat))
            objectness.append(conv_obj(reg_feat))
        return YOLOXOut(
            cls_score=cls_score, bbox_pred=bbox_pred, objectness=objectness
        )

    def __call__(self, features: list[torch.Tensor]) -> YOLOXOut:
        """Type definition for call implementation."""
        return self._call_impl(features)


def bboxes_nms(
    cls_scores: torch.Tensor,
    bboxes: torch.Tensor,
    objectness: torch.Tensor,
    nms_threshold: float = 0.7,
    score_thr: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode box energies into detections for a single image.

    Detections are post-processed via NMS. NMS is performed per level.
    Afterwards, select topk detections.

    Args:
        cls_scores (torch.Tensor): topk class scores per level.
        bboxes (torch.Tensor): topk class labels per level.
        objectness (torch.Tensor): topk regression params per level.
        nms_threshold (float, optional): iou threshold for NMS.
            Defaults to 0.7.
        score_thr (float, optional): score threshold to filter detections.
            Defaults to 0.0.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: decoded boxes, scores,
            and labels.
    """
    max_scores, labels = torch.max(cls_scores, 1)
    valid_mask = objectness * max_scores >= score_thr

    bboxes = bboxes[valid_mask]
    scores = max_scores[valid_mask] * objectness[valid_mask]
    labels = labels[valid_mask]

    if labels.numel() > 0:
        keep = batched_nms(bboxes, scores, labels, nms_threshold)
        return bboxes[keep], scores[keep], labels[keep]
    return bboxes.new_zeros(0, 4), scores.new_zeros(0), labels.new_zeros(0)


class YOLOXPostprocess(nn.Module):
    """Postprocess detections from YOLOX detection head.

    Args:
        point_generator (MlvlPointGenerator): Point generator.
        box_decoder (YOLOXBBoxDecoder): Box decoder.
        nms_threshold (float, optional): IoU threshold for NMS. Defaults to
            0.65.
        score_thr (float, optional): Score threshold to filter detections.
            Defaults to 0.01.
    """

    def __init__(
        self,
        point_generator: MlvlPointGenerator,
        box_decoder: YOLOXBBoxDecoder,
        nms_threshold: float = 0.65,
        score_thr: float = 0.01,
    ) -> None:
        """Creates an instance of the class."""
        super().__init__()
        self.point_generator = point_generator
        self.box_decoder = box_decoder
        self.nms_threshold = nms_threshold
        self.score_thr = score_thr

    def forward(
        self,
        cls_outs: list[torch.Tensor],
        reg_outs: list[torch.Tensor],
        obj_outs: list[torch.Tensor],
        images_hw: list[tuple[int, int]],
    ) -> DetOut:
        """Forward pass.

        Args:
            cls_outs (list[torch.Tensor]): [N, C, H, W] per scale.
            reg_outs (list[torch.Tensor]): [N, 4, H, W] per scale.
            obj_outs (list[torch.Tensor]): [N, 1, H, W] per scale.
            images_hw (list[tuple[int, int]]): list of image sizes.

        Returns:
            DetOut: Detection outputs.
        """
        dtype, device = cls_outs[0].dtype, cls_outs[0].device
        num_imgs = len(images_hw)
        num_classes = cls_outs[0].shape[1]
        featmap_sizes: list[tuple[int, int]] = [
            tuple(featmap.size()[-2:]) for featmap in cls_outs  # type: ignore
        ]
        assert len(featmap_sizes) == self.point_generator.num_levels
        mlvl_points = self.point_generator.grid_priors(
            featmap_sizes, dtype=dtype, device=device, with_stride=True
        )

        # flatten cls_outs, reg_outs and obj_outs
        cls_list = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, num_classes)
            for cls_score in cls_outs
        ]
        reg_list = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in reg_outs
        ]
        obj_list = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in obj_outs
        ]

        flatten_cls = torch.cat(cls_list, dim=1).sigmoid()
        flatten_reg = torch.cat(reg_list, dim=1)
        flatten_obj = torch.cat(obj_list, dim=1).sigmoid()
        flatten_points = torch.cat(mlvl_points)

        flatten_boxes = self.box_decoder(flatten_points, flatten_reg)

        bbox_list, score_list, label_list = [], [], []
        for img_id, _ in enumerate(images_hw):
            bboxes, scores, labels = bboxes_nms(
                flatten_cls[img_id],
                flatten_boxes[img_id],
                flatten_obj[img_id],
                nms_threshold=self.nms_threshold,
                score_thr=self.score_thr,
            )
            bbox_list.append(bboxes)
            score_list.append(scores)
            label_list.append(labels)
        return DetOut(bbox_list, score_list, label_list)

    def __call__(
        self,
        cls_outs: list[torch.Tensor],
        reg_outs: list[torch.Tensor],
        obj_outs: list[torch.Tensor],
        images_hw: list[tuple[int, int]],
    ) -> DetOut:
        """Type definition for function call."""
        return self._call_impl(cls_outs, reg_outs, obj_outs, images_hw)
