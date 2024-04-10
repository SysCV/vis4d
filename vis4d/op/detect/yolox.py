"""YOLOX detection head.

Modified from mmdetection (https://github.com/open-mmlab/mmdetection).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.ops import batched_nms

from vis4d.common import TorchLossFunc
from vis4d.common.distributed import reduce_mean
from vis4d.op.box.anchor import MlvlPointGenerator
from vis4d.op.box.encoder import YOLOXBBoxDecoder
from vis4d.op.box.matchers import SimOTAMatcher
from vis4d.op.box.samplers import PseudoSampler
from vis4d.op.layer import Conv2d
from vis4d.op.layer.weight_init import bias_init_with_prob
from vis4d.op.loss import IoULoss
from vis4d.op.loss.reducer import SumWeightedLoss

from .common import DetOut


class YOLOXOut(NamedTuple):
    """YOLOX head outputs."""

    # Logits for box classification for each feature level. The logit
    # dimention is [batch_size, number of classes, height, width].
    cls_score: list[torch.Tensor]
    # Each box has regression for all classes for each feature level. So the
    # tensor dimension is [batch_size, 4, height, width].
    bbox_pred: list[torch.Tensor]
    # Objectness scores for each feature level. The tensor dimension is
    # [batch_size, 1, height, width]
    objectness: list[torch.Tensor]


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
        self._init_weights()

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

    def _init_weights(self) -> None:
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(
                    m.weight,
                    a=math.sqrt(5),
                    mode="fan_in",
                    nonlinearity="leaky_relu",
                )
        bias_init = bias_init_with_prob(0.01)
        for conv_cls, conv_obj in zip(
            self.multi_level_conv_cls, self.multi_level_conv_obj
        ):
            conv_cls.bias.data.fill_(bias_init)
            conv_obj.bias.data.fill_(bias_init)

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
    nms_threshold: float = 0.65,
    score_thr: float = 0.01,
    nms_pre: int = -1,
    max_per_img: int = -1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode box energies into detections for a single image.

    Detections are post-processed via NMS. NMS is performed per level.
    Afterwards, select topk detections.

    Args:
        cls_scores (torch.Tensor): topk class scores per level.
        bboxes (torch.Tensor): topk class labels per level.
        objectness (torch.Tensor): topk regression params per level.
        nms_threshold (float, optional): iou threshold for NMS.
            Defaults to 0.65.
        score_thr (float, optional): score threshold to filter detections.
            Defaults to 0.01.
        nms_pre (int, optional): number of topk results before NMS.
            Defaults to -1 (all).
        max_per_img (int, optional): number of topk results after NMS.
            Defaults to -1 (all).

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: decoded boxes, scores,
            and labels.
    """
    if nms_pre == -1:
        nms_pre = len(cls_scores)
    if max_per_img == -1:
        max_per_img = len(cls_scores)
    max_scores, labels = torch.max(cls_scores, 1)
    valid_mask = objectness * max_scores >= score_thr
    valid_idxs = valid_mask.nonzero()[:, 0]
    num_topk = min(nms_pre, valid_mask.sum())  # type: ignore

    scores, idxs = (max_scores[valid_mask] * objectness[valid_mask]).sort(
        descending=True
    )
    scores = scores[:num_topk]
    topk_idxs = valid_idxs[idxs[:num_topk]]

    bboxes = bboxes[topk_idxs]
    labels = labels[topk_idxs]

    if labels.numel() > 0:
        keep = batched_nms(bboxes, scores, labels, nms_threshold)[:max_per_img]
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

    flatten_cls = torch.cat(cls_list, dim=1)
    flatten_reg = torch.cat(reg_list, dim=1)
    flatten_obj = torch.cat(obj_list, dim=1)
    flatten_points = torch.cat(mlvl_points)

    flatten_boxes = box_decoder(flatten_points, flatten_reg)
    return flatten_cls, flatten_reg, flatten_obj, flatten_points, flatten_boxes


class YOLOXPostprocess(nn.Module):
    """Postprocess detections from YOLOX detection head."""

    def __init__(
        self,
        point_generator: MlvlPointGenerator,
        box_decoder: YOLOXBBoxDecoder,
        nms_threshold: float = 0.65,
        score_thr: float = 0.01,
        nms_pre: int = -1,
        max_per_img: int = -1,
    ) -> None:
        """Creates an instance of the class.

        Args:
            point_generator (MlvlPointGenerator): Point generator.
            box_decoder (YOLOXBBoxDecoder): Box decoder.
            nms_threshold (float, optional): IoU threshold for NMS. Defaults to
                0.65.
            score_thr (float, optional): Score threshold to filter detections.
                Defaults to 0.01.
            nms_pre (int, optional): Number of topk results before NMS.
                Defaults to -1 (all).
            max_per_img (int, optional): Number of topk results after NMS.
                Defaults to -1 (all).
        """
        super().__init__()
        self.point_generator = point_generator
        self.box_decoder = box_decoder
        self.nms_threshold = nms_threshold
        self.score_thr = score_thr
        self.nms_pre = nms_pre
        self.max_per_img = max_per_img

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
        flatten_cls, flatten_obj = flatten_cls.sigmoid(), flatten_obj.sigmoid()

        bbox_list, score_list, label_list = [], [], []
        for img_id, _ in enumerate(images_hw):
            bboxes, scores, labels = bboxes_nms(
                flatten_cls[img_id],
                flatten_boxes[img_id],
                flatten_obj[img_id],
                nms_threshold=self.nms_threshold,
                score_thr=self.score_thr,
                nms_pre=self.nms_pre,
                max_per_img=self.max_per_img,
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
    loss_obj: Tensor
    loss_l1: Tensor | None


def bbox_xyxy_to_cxcywh(bbox: Tensor) -> Tensor:
    """Convert bbox coordinates from (x1, y1, x2, y2) to (cx, cy, w, h).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    x1, y1, x2, y2 = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)]
    return torch.cat(bbox_new, dim=-1)


def get_l1_target(
    bbox_target: Tensor, priors: Tensor, eps: float = 1e-8
) -> Tensor:
    """Convert gt bboxes to center offset and log width height.

    Args:
        bbox_target (Tensor): Shape (n, 4) for ground-truth bboxes.
        priors (Tensor): Shape (n, 4) for prior boxes.
        eps (float, optional): Epsilon for numerical stability. Defaults to
            1e-8.
    """
    l1_target = bbox_target.new_zeros((len(bbox_target), 4))
    gt_cxcywh = bbox_xyxy_to_cxcywh(bbox_target)
    l1_target[:, :2] = (gt_cxcywh[:, :2] - priors[:, :2]) / priors[:, 2:]
    l1_target[:, 2:] = torch.log(gt_cxcywh[:, 2:] / priors[:, 2:] + eps)
    return l1_target


class YOLOXHeadLoss(nn.Module):
    """Loss of YOLOX head."""

    def __init__(
        self,
        num_classes: int,
        point_generator: MlvlPointGenerator | None = None,
        box_decoder: YOLOXBBoxDecoder | None = None,
        loss_cls: TorchLossFunc = F.binary_cross_entropy_with_logits,
        loss_bbox: TorchLossFunc = IoULoss(mode="square", eps=1e-16),
        loss_obj: TorchLossFunc = F.binary_cross_entropy_with_logits,
        loss_l1: TorchLossFunc | None = None,
    ) -> None:
        """Creates an instance of the class.

        Args:
            num_classes (int): Number of classes.
            point_generator (MlvlPointGenerator): Point generator.
            box_decoder (YOLOXBBoxDecoder): Box decoder.
            loss_cls (TorchLossFunc, optional): Classification loss function.
                Defaults to sigmoid_focal_loss.
            loss_bbox (TorchLossFunc, optional): Regression loss function.
                Defaults to l1_loss.
            loss_obj (TorchLossFunc, optional): Objectness loss function.
                Defaults to sigmoid_focal_loss.
            loss_l1 (TorchLossFunc | None, optional): L1 loss function.
                Defaults to None. Only used during the final few epochs.
        """
        super().__init__()
        self.num_classes = num_classes
        self.point_generator = (
            point_generator
            if point_generator is not None
            else get_default_point_generator()
        )
        if box_decoder is None:
            self.box_decoder = YOLOXBBoxDecoder()
        else:
            self.box_decoder = box_decoder
        self.box_matcher = SimOTAMatcher()
        self.box_sampler = PseudoSampler()
        self.loss_cls = loss_cls
        self.loss_bbox = loss_bbox
        self.loss_obj = loss_obj
        self.loss_l1 = loss_l1

    def _get_target_single(
        self,
        cls_preds: Tensor,
        objectness: Tensor,
        priors: Tensor,
        decoded_bboxes: Tensor,
        gt_bboxes: Tensor,
        gt_labels: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, int]:
        """Compute YOLOX training targets in a single image.

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

        scores = cls_preds.sigmoid() * objectness.unsqueeze(1).sigmoid()
        match_result = self.box_matcher(
            scores.sqrt_(),
            offset_priors,
            decoded_bboxes,
            gt_bboxes,
            gt_labels,
        )
        sampling_result = self.box_sampler(match_result)
        positives = sampling_result.sampled_labels == 1
        pos_inds = sampling_result.sampled_box_indices[positives]
        pos_tgt_inds = sampling_result.sampled_target_indices[positives]
        num_pos_per_img = pos_inds.size(0)

        pos_ious = match_result.assigned_gt_iou[pos_inds]
        # IOU aware classification score
        cls_target = F.one_hot(  # pylint: disable=not-callable
            gt_labels[pos_tgt_inds], self.num_classes
        ) * pos_ious.unsqueeze(-1)
        obj_target = torch.zeros_like(objectness).unsqueeze(-1)
        obj_target[pos_inds] = 1
        bbox_target = gt_bboxes[pos_tgt_inds]
        if self.loss_l1 is not None:
            l1_target = get_l1_target(bbox_target, priors[pos_inds])
        else:
            l1_target = bbox_target.new_zeros((len(bbox_target), 4))
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

    def forward(
        self,
        cls_outs: list[Tensor],
        reg_outs: list[Tensor],
        obj_outs: list[Tensor],
        target_boxes: list[Tensor],
        target_class_ids: list[Tensor],
        images_hw: list[tuple[int, int]],
    ) -> YOLOXHeadLosses:
        """Compute YOLOX classification, regression, and objectness losses.

        Args:
            cls_outs (list[Tensor]): Network classification outputs at all
                scales.
            reg_outs (list[Tensor]): Network regression outputs at all scales.
            obj_outs (list[Tensor]): Network objectness outputs at all scales.
            target_boxes (list[Tensor]): Target bounding boxes.
            images_hw (list[tuple[int, int]]): Image dimensions without
                padding.
            target_class_ids (list[Tensor]): Target class labels.

        Returns:
            YOLOXHeadLosses: YOLOX losses.
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

        num_imgs = len(images_hw)
        pos_masks_list, cls_targets_list, obj_targets_list = [], [], []
        bbox_targets_list, l1_targets_list, num_fg_imgs_list = [], [], []
        for flat_cls, flat_obj, flat_pts, flat_bxs, tgt_bxs, tgt_cls in zip(
            flatten_cls.detach(),
            flatten_obj.detach(),
            flatten_points.unsqueeze(0).repeat(num_imgs, 1, 1),
            flatten_boxes.detach(),
            target_boxes,
            target_class_ids,
        ):
            targets = self._get_target_single(
                flat_cls, flat_obj, flat_pts, flat_bxs, tgt_bxs, tgt_cls
            )
            pos_masks_list.append(targets[0])
            cls_targets_list.append(targets[1])
            obj_targets_list.append(targets[2])
            bbox_targets_list.append(targets[3])
            l1_targets_list.append(targets[4])
            num_fg_imgs_list.append(targets[5])

        num_pos = torch.tensor(
            sum(num_fg_imgs_list), dtype=torch.float, device=flatten_cls.device
        )
        num_total_samples: Tensor | float = max(  # type: ignore
            reduce_mean(num_pos), 1.0
        )

        pos_masks = torch.cat(pos_masks_list, 0)
        cls_targets = torch.cat(cls_targets_list, 0)
        obj_targets = torch.cat(obj_targets_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        if self.loss_l1 is not None:
            l1_targets = torch.cat(l1_targets_list, 0)

        loss_obj = self.loss_obj(
            flatten_obj.view(-1, 1), obj_targets, reduction="none"
        )
        loss_obj = SumWeightedLoss(1.0, num_total_samples)(loss_obj)

        if num_pos > 0:
            loss_cls = self.loss_cls(
                flatten_cls.view(-1, self.num_classes)[pos_masks],
                cls_targets,
                reduction="none",
            )
            loss_cls = SumWeightedLoss(1.0, num_total_samples)(loss_cls)
            loss_bbox = self.loss_bbox(
                flatten_boxes.view(-1, 4)[pos_masks], bbox_targets
            )
            loss_bbox = SumWeightedLoss(5.0, num_total_samples)(loss_bbox)
        else:
            loss_cls = flatten_cls.sum() * 0
            loss_bbox = flatten_boxes.sum() * 0

        if self.loss_l1 is not None:
            if num_pos > 0:
                loss_l1 = self.loss_l1(
                    flatten_reg.view(-1, 4)[pos_masks], l1_targets
                )
                loss_l1 = SumWeightedLoss(1.0, num_total_samples)(loss_l1)
            else:
                loss_l1 = flatten_reg.sum() * 0
        else:
            loss_l1 = None

        return YOLOXHeadLosses(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_obj=loss_obj,
            loss_l1=loss_l1,
        )

    def __call__(
        self,
        cls_outs: list[Tensor],
        reg_outs: list[Tensor],
        obj_outs: list[Tensor],
        target_boxes: list[Tensor],
        target_class_ids: list[Tensor],
        images_hw: list[tuple[int, int]],
    ) -> YOLOXHeadLosses:
        """Type definition."""
        return self._call_impl(
            cls_outs,
            reg_outs,
            obj_outs,
            target_boxes,
            target_class_ids,
            images_hw,
        )
