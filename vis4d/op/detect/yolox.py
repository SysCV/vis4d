"""YOLOX detection head.

Modified from mmdetection (https://github.com/open-mmlab/mmdetection).
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torchvision.ops import batched_nms

from vis4d.common import TorchLossFunc
from vis4d.op.box.anchor import MlvlPointGenerator
from vis4d.op.box.encoder import YOLOXBBoxDecoder
from vis4d.op.box.matchers import Matcher, SimOTAMatcher
from vis4d.op.box.samplers import PseudoSampler
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


def preprocess_outputs(
    cls_outs: list[torch.Tensor],
    reg_outs: list[torch.Tensor],
    obj_outs: list[torch.Tensor],
    images_hw: list[tuple[int, int]],
    point_generator: MlvlPointGenerator,
    box_decoder: YOLOXBBoxDecoder,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Preprocess model outputs before postprocessing/loss computation.

    Args:
        cls_outs (list[torch.Tensor]): [N, C, H, W] per scale.
        reg_outs (list[torch.Tensor]): [N, 4, H, W] per scale.
        obj_outs (list[torch.Tensor]): [N, 1, H, W] per scale.
        images_hw (list[tuple[int, int]]): List of image sizes.
        point_generator (MlvlPointGenerator): Point generator.
        box_decoder (YOLOXBBoxDecoder): Box decoder.

    Returns:
        tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: Flattened outputs.
    """
    dtype, device = cls_outs[0].dtype, cls_outs[0].device
    num_imgs = len(images_hw)
    num_classes = cls_outs[0].shape[1]
    featmap_sizes: list[tuple[int, int]] = [
        tuple(featmap.size()[-2:]) for featmap in cls_outs  # type: ignore
    ]
    assert len(featmap_sizes) == point_generator.num_levels
    mlvl_points = point_generator.grid_priors(
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

    flatten_boxes = box_decoder(flatten_points, flatten_reg)
    return flatten_cls, flatten_reg, flatten_obj, flatten_points, flatten_boxes


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
        flatten_cls, _, flatten_obj, _, flatten_boxes = preprocess_outputs(
            cls_outs,
            reg_outs,
            obj_outs,
            images_hw,
            self.point_generator,
            self.box_decoder,
        )

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


class YOLOXHeadLosses(NamedTuple):
    """YOLOX head loss container."""

    loss_cls: Tensor
    loss_bbox: Tensor


def _get_target_single(
    self, cls_preds, objectness, priors, decoded_bboxes, gt_bboxes, gt_labels
):
    """Compute classification, regression, and objectness targets for
    priors in a single image.
    Args:
        cls_preds (Tensor): Classification predictions of one image,
            a 2D-Tensor with shape [num_priors, num_classes]
        objectness (Tensor): Objectness predictions of one image,
            a 1D-Tensor with shape [num_priors]
        priors (Tensor): All priors of one image, a 2D-Tensor with shape
            [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
        decoded_bboxes (Tensor): Decoded bboxes predictions of one image,
            a 2D-Tensor with shape [num_priors, 4] in [tl_x, tl_y,
            br_x, br_y] format.
        gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
            with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
        gt_labels (Tensor): Ground truth labels of one image, a Tensor
            with shape [num_gts].
    """

    num_priors = priors.size(0)
    num_gts = gt_labels.size(0)
    gt_bboxes = gt_bboxes.to(decoded_bboxes.dtype)
    # No target
    if num_gts == 0:
        cls_target = cls_preds.new_zeros((0, self.num_classes))
        bbox_target = cls_preds.new_zeros((0, 4))
        l1_target = cls_preds.new_zeros((0, 4))
        obj_target = cls_preds.new_zeros((num_priors, 1))
        foreground_mask = cls_preds.new_zeros(num_priors).bool()
        return (
            foreground_mask,
            cls_target,
            obj_target,
            bbox_target,
            l1_target,
            0,
        )

    # YOLOX uses center priors with 0.5 offset to assign targets,
    # but use center priors without offset to regress bboxes.
    offset_priors = torch.cat(
        [priors[:, :2] + priors[:, 2:] * 0.5, priors[:, 2:]], dim=-1
    )

    assign_result = self.assigner.assign(
        cls_preds.sigmoid() * objectness.unsqueeze(1).sigmoid(),
        offset_priors,
        decoded_bboxes,
        gt_bboxes,
        gt_labels,
    )

    sampling_result = self.sampler.sample(assign_result, priors, gt_bboxes)
    pos_inds = sampling_result.pos_inds
    num_pos_per_img = pos_inds.size(0)

    pos_ious = assign_result.max_overlaps[pos_inds]
    # IOU aware classification score
    cls_target = F.one_hot(
        sampling_result.pos_gt_labels, self.num_classes
    ) * pos_ious.unsqueeze(-1)
    obj_target = torch.zeros_like(objectness).unsqueeze(-1)
    obj_target[pos_inds] = 1
    bbox_target = sampling_result.pos_gt_bboxes
    l1_target = cls_preds.new_zeros((num_pos_per_img, 4))
    if self.use_l1:
        l1_target = get_l1_target(l1_target, bbox_target, priors[pos_inds])
    foreground_mask = torch.zeros_like(objectness).to(torch.bool)
    foreground_mask[pos_inds] = 1
    return (
        foreground_mask,
        cls_target,
        obj_target,
        bbox_target,
        l1_target,
        num_pos_per_img,
    )


def get_l1_target(l1_target, gt_bboxes, priors, eps=1e-8):
    """Convert gt bboxes to center offset and log width height."""
    gt_cxcywh = bbox_xyxy_to_cxcywh(gt_bboxes)
    l1_target[:, :2] = (gt_cxcywh[:, :2] - priors[:, :2]) / priors[:, 2:]
    l1_target[:, 2:] = torch.log(gt_cxcywh[:, 2:] / priors[:, 2:] + eps)
    return l1_target


class YOLOXHeadLoss(nn.Module):
    """Loss of YOLOX head."""

    def __init__(
        self,
        point_generator: MlvlPointGenerator,
        box_decoder: YOLOXBBoxDecoder,
        loss_cls: TorchLossFunc = sigmoid_focal_loss,
        loss_bbox: TorchLossFunc = l1_loss,
        loss_obj: TorchLossFunc = l1_loss,
        loss_l1: TorchLossFunc = l1_loss,
    ) -> None:
        """Creates an instance of the class.

        Args:
            point_generator (MlvlPointGenerator): Point generator.
            box_decoder (YOLOXBBoxDecoder): Box decoder.
            loss_cls (TorchLossFunc, optional): Classification loss function.
                Defaults to sigmoid_focal_loss.
            loss_bbox (TorchLossFunc, optional): Regression loss function.
                Defaults to l1_loss.
        """
        super().__init__()
        self.point_generator = point_generator
        self.box_decoder = box_decoder
        self.matcher = SimOTAMatcher()
        self.box_sampler = PseudoSampler()
        self.loss_cls = loss_cls
        self.loss_bbox = loss_bbox
        self.loss_obj = loss_obj
        self.loss_l1 = loss_l1

    def forward(
        self,
        cls_outs: list[Tensor],
        reg_outs: list[Tensor],
        obj_outs: list[Tensor],
        target_boxes: list[Tensor],
        images_hw: list[tuple[int, int]],
        target_class_ids: list[Tensor | float] | None = None,
    ) -> YOLOXHeadLosses:
        """Compute RetinaNet classification and regression losses.

        Args:
            cls_outs (list[Tensor]): Network classification outputs at all
                scales.
            reg_outs (list[Tensor]): Network regression outputs at all scales.
            obj_outs (list[Tensor]): Network objectness outputs at all scales.
            target_boxes (list[Tensor]): Target bounding boxes.
            images_hw (list[tuple[int, int]]): Image dimensions without
                padding.
            target_class_ids (list[Tensor] | None, optional): Target
                class labels.

        Returns:
            DenseAnchorHeadLosses: Classification and regression losses.
        """
        (
            flatten_cls,
            flatten_reg,
            flatten_obj,
            flatten_points,
            flatten_boxes,
        ) = preprocess_outputs(
            cls_outs,
            reg_outs,
            obj_outs,
            images_hw,
            self.point_generator,
            self.box_decoder,
        )

        targets_per_level, num_samples = get_targets_per_batch(
            featmap_sizes,
            target_boxes,
            target_class_ids,
            images_hw,
            self.anchor_generator,
            self.box_encoder,
            self.matcher,
            self.sampler,
            self.allowed_border,
        )

        device = cls_outs[0].device
        loss_cls_all = torch.tensor(0.0, device=device)
        loss_bbox_all = torch.tensor(0.0, device=device)
        for level_id, (cls_out, reg_out) in enumerate(zip(cls_outs, reg_outs)):
            box_tgt, box_wgt, lbl, lbl_wgt = targets_per_level[level_id]
            loss_cls, loss_bbox = self._loss_single_scale(
                cls_out, reg_out, box_tgt, box_wgt, lbl, lbl_wgt, num_samples
            )
            loss_cls_all += loss_cls
            loss_bbox_all += loss_bbox
        return YOLOXHeadLosses(loss_cls=loss_cls_all, loss_bbox=loss_bbox_all)

    def __call__(
        self,
        cls_outs: list[Tensor],
        reg_outs: list[Tensor],
        target_boxes: list[Tensor],
        images_hw: list[tuple[int, int]],
        target_class_ids: list[Tensor] | None = None,
    ) -> YOLOXHeadLosses:
        """Type definition."""
        return self._call_impl(
            cls_outs, reg_outs, target_boxes, images_hw, target_class_ids
        )
