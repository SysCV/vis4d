"""RetinaNet."""
from math import prod
from typing import List, NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import batched_nms

from vis4d.op.box.encoder import BaseBoxEncoder2D, DeltaXYWHBBoxEncoder
from vis4d.op.box.matchers import BaseMatcher, MaxIoUMatcher
from vis4d.op.box.samplers import BaseSampler, PseudoSampler
from vis4d.op.detect.rpn import images_to_levels, unmap
from vis4d.op.loss.common import l1_loss
from vis4d.op.loss.reducer import SumWeightedLoss
from vis4d.struct_to_revise.labels.boxes import filter_boxes

from .anchor_generator import AnchorGenerator, anchor_inside_image
from .rcnn import DetOut


class RetinaNetOut(NamedTuple):
    """RetinaNet head outputs."""

    # logits for the box classication for each feature level. The logit
    # dimention is number of classes plus 1 for the background.
    cls_score: List[torch.Tensor]
    # Each box has regression for all classes for each feature level. So the
    # tensor dimention is [batch_size, number of boxes, number of classes x 4]
    bbox_pred: List[torch.Tensor]


def get_default_anchor_generator() -> AnchorGenerator:
    """Get default anchor generator."""
    return AnchorGenerator(
        octave_base_scale=4,
        scales_per_octave=3,
        ratios=[0.5, 1.0, 2.0],
        strides=[8, 16, 32, 64, 128],
    )


def get_default_box_encoder() -> DeltaXYWHBBoxEncoder:
    """Get the default bounding box encoder."""
    return DeltaXYWHBBoxEncoder(
        target_means=(0.0, 0.0, 0.0, 0.0), target_stds=(1.0, 1.0, 1.0, 1.0)
    )


def get_default_box_matcher() -> MaxIoUMatcher:
    """Get default bounding box matcher."""
    return MaxIoUMatcher(
        thresholds=[0.4, 0.5],
        labels=[0, -1, 1],
        allow_low_quality_matches=True,
    )


def get_default_box_sampler() -> PseudoSampler:
    """Get default bounding box sampler."""
    return PseudoSampler(0, 0)


class RetinaNetHead(nn.Module):
    """RetinaNet Head."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        feat_channels: int = 256,
        stacked_convs: int = 4,
        use_sigmoid_cls: bool = True,
        anchor_generator: Optional[AnchorGenerator] = None,
        box_encoder: Optional[BaseBoxEncoder2D] = None,
        box_matcher: Optional[BaseMatcher] = None,
    ):
        """Init."""
        super().__init__()
        self.anchor_generator = (
            anchor_generator
            if anchor_generator is not None
            else get_default_anchor_generator()
        )
        self.box_encoder = (
            box_encoder
            if box_encoder is not None
            else get_default_box_encoder()
        )
        self.box_matcher = (
            box_matcher
            if box_matcher is not None
            else get_default_box_matcher()
        )
        num_base_priors = self.anchor_generator.num_base_priors[0]

        if use_sigmoid_cls:
            cls_out_channels = num_classes
        else:
            cls_out_channels = num_classes + 1
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(stacked_convs):
            chn = in_channels if i == 0 else feat_channels
            self.cls_convs.append(
                nn.Conv2d(chn, feat_channels, 3, stride=1, padding=1),
            )
            self.reg_convs.append(
                nn.Conv2d(chn, feat_channels, 3, stride=1, padding=1),
            )
        self.retina_cls = nn.Conv2d(
            feat_channels, num_base_priors * cls_out_channels, 3, padding=1
        )
        self.retina_reg = nn.Conv2d(
            feat_channels, num_base_priors * 4, 3, padding=1
        )

    def forward(self, features: List[torch.Tensor]) -> RetinaNetOut:
        """RetinaNet forward.

        Args:
            features (List[torch.Tensor]): Feature pyramid

        Returns:
            RetinaNetOut: classification score and box prediction.
        """
        cls_scores, bbox_preds = [], []
        for feat in features:
            cls_feat = feat
            reg_feat = feat
            for cls_conv in self.cls_convs:
                cls_feat = self.relu(cls_conv(cls_feat))
            for reg_conv in self.reg_convs:
                reg_feat = self.relu(reg_conv(reg_feat))
            cls_scores.append(self.retina_cls(cls_feat))
            bbox_preds.append(self.retina_reg(reg_feat))
        return RetinaNetOut(cls_score=cls_scores, bbox_pred=bbox_preds)

    def __call__(self, features: List[torch.Tensor]) -> RetinaNetOut:
        """Type definition for call implementation."""
        return self._call_impl(features)


def get_params_per_level(
    cls_out: torch.Tensor,
    reg_out: torch.Tensor,
    anchors: torch.Tensor,
    num_pre_nms: int = 2000,
    score_thr: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get a topk pre-selection of flattened classification scores and box
    energies from feature output per level per image before nms.

    Args:
        cls_out (torch.Tensor): [C, H, W] classification scores at a particular scale.
        reg_out (torch.Tensor): [C, H, W] regression parameters at a particular scale.
        anchors (torch.Tensor): [H*W, 4] anchor boxes per cell.
        num_pre_nms (int): number of predictions before nms.
        score_thr (float): score threshold for filtering predictions.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: topk
            flattened classification, regression outputs, and corresponding anchors.
    """
    assert cls_out.size()[-2:] == reg_out.size()[-2:], (
        f"Shape mismatch: cls_out({cls_out.size()[-2:]}), reg_out("
        f"{reg_out.size()[-2:]})."
    )
    reg_out = reg_out.permute(1, 2, 0).reshape(-1, 4)
    cls_out = cls_out.permute(1, 2, 0).reshape(reg_out.size(0), -1).sigmoid()
    valid_mask = cls_out > score_thr
    valid_idxs = torch.nonzero(valid_mask)
    num_topk = min(num_pre_nms, valid_idxs.size(0))
    cls_out_filt = cls_out[valid_mask]
    cls_out_ranked, rank_inds = cls_out_filt.sort(descending=True)
    topk_inds = valid_idxs[rank_inds[:num_topk]]
    keep_inds, labels = topk_inds.unbind(dim=1)
    cls_out = cls_out_ranked[:num_topk]
    reg_out = reg_out[keep_inds, :]
    anchors = anchors[keep_inds, :]

    return cls_out, labels, reg_out, anchors


def decode_multi_level_outputs(
    cls_out_all: List[torch.Tensor],
    lbl_out_all: List[torch.Tensor],
    reg_out_all: List[torch.Tensor],
    anchors_all: List[torch.Tensor],
    image_hw: Tuple[int, int],
    box_encoder: DeltaXYWHBBoxEncoder,
    max_per_img: int = 1000,
    nms_threshold: float = 0.7,
    min_box_size: Tuple[int, int] = (0, 0),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode box energies into detections for a single image, post-process
    via NMS. NMS is performed per level. Afterwards, select topk detections.

    Args:
        cls_out_all (List[torch.Tensor]): topk class scores per level.
        lbl_out_all (List[torch.Tensor]): topk class labels per level.
        reg_out_all (List[torch.Tensor]): topk regression params per level.
        anchors_all (List[torch.Tensor]): topk anchor boxes per level.
        image_hw (Tuple[int, int]): image size.
        box_encoder (DeltaXYWHBBoxEncoder): bounding box encoder.
        max_per_img (int, optional): maximum predictions per image. Defaults to 1000.
        nms_threshold (float, optional): iou threshold for NMS. Defaults to 0.7.
        min_box_size (Tuple[int, int], optional): minimum box size. Defaults to (0, 0).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: decoded proposal boxes & scores.
    """
    scores, labels = torch.cat(cls_out_all), torch.cat(lbl_out_all)
    boxes = box_encoder.decode(
        torch.cat(anchors_all), torch.cat(reg_out_all), max_shape=image_hw
    )

    boxes, mask = filter_boxes(boxes, min_area=prod(min_box_size))
    scores, labels = scores[mask], labels[mask]

    if boxes.numel() > 0:
        keep = batched_nms(boxes, scores, labels, iou_threshold=nms_threshold)[
            :max_per_img
        ]
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    else:
        return (
            boxes.new_zeros(0, 4),
            scores.new_zeros(0),
            labels.new_zeros(0),
        )
    return boxes, scores, labels


class Dense2Det(nn.Module):
    """Compute detections from dense network outputs.

    This class acts as a stateless functor that does the following:
    1. Create anchor grid for feature grids (classification and regression outputs) at all scales.
    For each image
        For each level
            2. Get a topk pre-selection of flattened classification scores and box energies from feature output before NMS.
        3. Decode class scores and box energies into detection boxes, apply NMS.
    Return detection boxes for all images.
    """

    def __init__(
        self,
        anchor_generator: AnchorGenerator,
        box_encoder: DeltaXYWHBBoxEncoder,
        num_pre_nms: int = 2000,
        max_per_img: int = 1000,
        nms_threshold: float = 0.7,
        min_box_size: Tuple[int, int] = (0, 0),
        score_thr: float = 0.0,
    ) -> None:
        """Init."""
        super().__init__()
        self.anchor_generator = anchor_generator
        self.box_encoder = box_encoder
        self.num_pre_nms = num_pre_nms
        self.max_per_img = max_per_img
        self.nms_threshold = nms_threshold
        self.min_box_size = min_box_size
        self.score_thr = score_thr

    def forward(
        self,
        class_outs: List[torch.Tensor],
        regression_outs: List[torch.Tensor],
        images_hw: List[Tuple[int, int]],
    ) -> DetOut:
        """Compute detections from dense network outputs.

        Generate anchor grid for all scales.
        For each batch element:
            Compute classification, regression and anchor pairs for all scales.
            Decode those pairs into proposals, post-process with NMS.

        Args:
            class_outs (List[torch.Tensor]): [N, 1 * A, H, W] per scale.
            regression_outs (List[torch.Tensor]): [N, 4 * A, H, W] per scale.
            images_hw (List[Tuple[int, int]]): list of image sizes.

        Returns:
            DetOut: detection outputs.
        """
        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        device = class_outs[0].device
        featmap_sizes = [featmap.size()[-2:] for featmap in class_outs]
        assert len(featmap_sizes) == self.anchor_generator.num_levels
        anchor_grids = self.anchor_generator.grid_priors(
            featmap_sizes, device=device
        )
        proposals, scores, labels = [], [], []
        for img_id, image_hw in enumerate(images_hw):
            cls_out_all, lbl_out_all, reg_out_all, anchors_all = [], [], [], []
            for cls_outs, reg_outs, anchor_grid in zip(
                class_outs, regression_outs, anchor_grids
            ):
                (cls_out, lbl_out, reg_out, anchors,) = get_params_per_level(
                    cls_outs[img_id],
                    reg_outs[img_id],
                    anchor_grid,
                    self.num_pre_nms,
                    self.score_thr,
                )
                cls_out_all += [cls_out]
                lbl_out_all += [lbl_out]
                reg_out_all += [reg_out]
                anchors_all += [anchors]

            box, score, label = decode_multi_level_outputs(
                cls_out_all,
                lbl_out_all,
                reg_out_all,
                anchors_all,
                image_hw,
                self.box_encoder,
                self.max_per_img,
                self.nms_threshold,
                self.min_box_size,
            )
            proposals.append(box)
            scores.append(score)
            labels.append(label)
        return DetOut(proposals, scores, labels)

    def __call__(
        self,
        class_outs: List[torch.Tensor],
        regression_outs: List[torch.Tensor],
        images_hw: List[Tuple[int, int]],
    ) -> DetOut:
        """Type definition for function call."""
        return self._call_impl(class_outs, regression_outs, images_hw)


class RetinaNetTargets(NamedTuple):
    """Targets for RetinaNetLoss."""

    labels: torch.Tensor
    label_weights: torch.Tensor
    bbox_targets: torch.Tensor
    bbox_weights: torch.Tensor


class RetinaNetLosses(NamedTuple):
    """RetinaNet loss container."""

    loss_cls: torch.Tensor
    loss_bbox: torch.Tensor


class RetinaNetLoss(nn.Module):
    """Loss of RetinaNet.

    For a given set of multi-scale dense outputs, compute the desired target
    outputs and apply classification and regression losses.
    The targets are computed with the given target bounding boxes, the
    anchor grid defined by the anchor generator and the given box encoder.
    """

    def __init__(
        self,
        anchor_generator: AnchorGenerator,
        box_encoder: DeltaXYWHBBoxEncoder,
        box_matcher: Optional[BaseMatcher] = None,
        box_sampler: Optional[BaseSampler] = None,
        loss_cls=None,
    ):
        """Init.

        Args:
            anchor_generator (AnchorGenerator): Generates anchor grid priors.
            box_encoder (DeltaXYWHBBoxEncoder): Encodes bounding boxes to
                the desired network output.
            box_matcher (Optional[BaseMatcher], optional): Box matcher.
            box_sampler (Optional[BaseSampler], optional): Box sampler.
            loss_cls: Classification loss.
        """
        super().__init__()
        self.anchor_generator = anchor_generator
        self.box_encoder = box_encoder
        self.allowed_border = 0
        self.matcher = (
            box_matcher
            if box_matcher is not None
            else get_default_box_matcher()
        )
        self.sampler = (
            box_sampler
            if box_sampler is not None
            else get_default_box_sampler()
        )
        self.loss_cls = (
            loss_cls
            if loss_cls is not None
            else F.binary_cross_entropy_with_logits
        )

    def _loss_single_scale(
        self,
        cls_out: torch.Tensor,
        reg_out: torch.Tensor,
        bbox_targets: torch.Tensor,
        bbox_weights: torch.Tensor,
        labels: torch.Tensor,
        label_weights: torch.Tensor,
        num_total_samples: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute losses per scale, all batch elements.

        Args:
            cls_out (torch.Tensor): [N, C, H, W] tensor of class logits.
            reg_out (torch.Tensor): [N, C, H, W] tensor of regression params.
            bbox_targets (torch.Tensor): [H*W, 4] bounding box targets
            bbox_weights (torch.Tensor): [H*W] per-sample weighting for loss.
            labels (torch.Tensor): [H*W] classification targets.
            label_weights (torch.Tensor): [H*W] per-sample weighting for loss.
            num_total_samples (int): average factor of loss.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: classification and regression
                losses.
        """
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_out.permute(0, 2, 3, 1).reshape(labels.size(0), -1)
        if cls_score.size(1) > 1:
            labels = F.one_hot(
                labels.long(), num_classes=cls_score.size(1) + 1
            )[:, : cls_score.size(1)].float()
            label_weights = label_weights.repeat(8).reshape(
                -1, cls_score.size(1)
            )
        loss_cls = self.loss_cls(cls_score, labels, reduction="none")
        loss_cls = SumWeightedLoss(label_weights, num_total_samples)(loss_cls)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = reg_out.permute(0, 2, 3, 1).reshape(-1, 4)
        loss_bbox = l1_loss(
            bbox_pred,
            bbox_targets,
            SumWeightedLoss(bbox_weights, num_total_samples),
        )
        return loss_cls, loss_bbox

    def _get_targets_per_image(
        self,
        target_boxes: torch.Tensor,
        anchors: torch.Tensor,
        image_hw: Tuple[int, int],
        target_class: Optional[torch.Tensor] = None,
    ) -> Tuple[RetinaNetTargets, int, int]:
        """Get targets per batch element, all scales."""
        inside_flags = anchor_inside_image(
            anchors, image_hw, allowed_border=self.allowed_border
        )
        # assign gt and sample anchors
        anchors = anchors[inside_flags, :]

        matching = self.matcher(anchors, target_boxes)
        sampling_result = self.sampler(matching)

        num_valid_anchors = anchors.size(0)
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_zeros((num_valid_anchors,))
        label_weights = anchors.new_zeros(num_valid_anchors)

        positives = sampling_result.sampled_labels == 1
        negatives = sampling_result.sampled_labels == 0
        pos_inds = sampling_result.sampled_box_indices[positives]
        pos_target_inds = sampling_result.sampled_target_indices[positives]
        neg_inds = sampling_result.sampled_box_indices[negatives]
        if len(pos_inds) > 0:
            pos_bbox_targets = self.box_encoder.encode(
                anchors[pos_inds],
                target_boxes[pos_target_inds],
            )
            bbox_targets[pos_inds] = pos_bbox_targets
            bbox_weights[pos_inds] = 1.0
            if target_class is None:
                # RPN
                labels[pos_inds] = 1.0
            else:
                labels[pos_inds] = target_class[pos_target_inds].float()
            label_weights[pos_inds] = 1.0
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        num_total_anchors = inside_flags.size(0)
        labels = unmap(labels, num_total_anchors, inside_flags)
        label_weights = unmap(label_weights, num_total_anchors, inside_flags)
        bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
        bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (
            RetinaNetTargets(
                labels, label_weights, bbox_targets, bbox_weights
            ),
            positives.sum(),
            negatives.sum(),
        )

    def forward(
        self,
        class_outs: List[torch.Tensor],
        regression_outs: List[torch.Tensor],
        target_boxes: List[torch.Tensor],
        images_hw: List[Tuple[int, int]],
        target_class_ids: Optional[List[torch.Tensor]] = None,
    ) -> RetinaNetLosses:
        """Compute RetinaNet classification and regression losses.

        Args:
            class_outs (List[torch.Tensor]): Network classification outputs at all scales.
            regression_outs (List[torch.Tensor]): Network regression outputs at all scales.
            target_boxes (List[torch.Tensor]): Target bounding boxes.
            images_hw (List[Tuple[int, int]]): Image dimensions without padding.
            target_class_ids (Optional[List[torch.Tensor]], optional): Target class labels.

        Returns:
            RetinaNetLosses: classification and regression losses.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in class_outs]
        assert len(featmap_sizes) == self.anchor_generator.num_levels
        if target_class_ids is None:
            target_class_ids = [None for _ in len(target_boxes)]

        device = class_outs[0].device

        anchor_grids = self.anchor_generator.grid_priors(
            featmap_sizes, device=device
        )
        num_level_anchors = [anchors.size(0) for anchors in anchor_grids]
        anchors_all_levels = torch.cat(anchor_grids)

        targets, num_total_pos, num_total_neg = [], 0, 0
        for tgt_box, tgt_cls, image_hw in zip(
            target_boxes, target_class_ids, images_hw
        ):
            target, num_pos, num_neg = self._get_targets_per_image(
                tgt_box, anchors_all_levels, image_hw, tgt_cls
            )
            num_total_pos += num_pos
            num_total_neg += num_neg
            bbox_targets_per_level = target.bbox_targets.split(
                num_level_anchors
            )
            bbox_weights_per_level = target.bbox_weights.split(
                num_level_anchors
            )
            labels_per_level = target.labels.split(num_level_anchors)
            label_weights_per_level = target.label_weights.split(
                num_level_anchors
            )
            targets.append(
                (
                    bbox_targets_per_level,
                    bbox_weights_per_level,
                    labels_per_level,
                    label_weights_per_level,
                )
            )
        targets_per_level = images_to_levels(targets)
        num_samples = num_total_pos + num_total_neg

        loss_cls_all = torch.tensor(0.0, device=device)
        loss_bbox_all = torch.tensor(0.0, device=device)
        for level_id, (cls_out, reg_out) in enumerate(
            zip(class_outs, regression_outs)
        ):
            loss_cls, loss_bbox = self._loss_single_scale(
                cls_out, reg_out, *targets_per_level[level_id], num_samples
            )
            loss_cls_all += loss_cls
            loss_bbox_all += loss_bbox
        return RetinaNetLosses(loss_cls=loss_cls_all, loss_bbox=loss_bbox_all)

    def __call__(
        self,
        class_outs: List[torch.Tensor],
        regression_outs: List[torch.Tensor],
        target_boxes: List[torch.Tensor],
        images_hw: List[Tuple[int, int]],
        target_class_ids: Optional[List[torch.Tensor]] = None,
    ) -> RetinaNetLosses:
        """Type definition."""
        return self._call_impl(
            class_outs,
            regression_outs,
            target_boxes,
            images_hw,
            target_class_ids,
        )
