"""Faster RCNN roi head."""
from math import prod
from typing import List, NamedTuple, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from vis4d.common.bbox.coders.delta_xywh_coder import DeltaXYWHBBoxEncoder
from vis4d.common.bbox.poolers import MultiScaleRoIAlign
from vis4d.common.bbox.utils import multiclass_nms
from vis4d.op.losses.utils import l1_loss, weight_reduce_loss
from vis4d.op.utils import segmentations_from_mmdet
from vis4d.struct import Detections


class RCNNOut(NamedTuple):
    cls_score: torch.Tensor
    bbox_pred: torch.Tensor


class RCNNHead(nn.Module):
    """faster rcnn box2d roi head."""

    def __init__(
        self,
        num_classes: int = 80,
        roi_size: Tuple[int, int] = (7, 7),
        in_channels: int = 256,
        fc_out_channels: int = 1024,
    ) -> None:
        """Init."""
        super().__init__()
        in_channels *= prod(roi_size)
        self.shared_fcs = nn.Sequential(
            nn.Linear(in_channels, fc_out_channels),
            nn.Linear(fc_out_channels, fc_out_channels),
        )

        self.roi_pooler = MultiScaleRoIAlign(
            sampling_ratio=0, resolution=(7, 7), strides=[4, 8, 16, 32]
        )
        self.fc_cls = nn.Linear(
            in_features=fc_out_channels, out_features=num_classes + 1
        )
        self.fc_reg = nn.Linear(
            in_features=fc_out_channels, out_features=4 * num_classes
        )
        self.relu = nn.ReLU(inplace=True)

        self._init_weights(self.fc_cls)
        self._init_weights(self.fc_reg, std=0.001)

    def _init_weights(
        self, module, std: float = 0.01
    ):  # TODO make this a common function?
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        features: List[torch.Tensor],
        boxes: List[torch.Tensor],
    ) -> RCNNOut:
        """Forward pass during training stage."""
        bbox_feats = self.roi_pooler(features, boxes).flatten(start_dim=1)
        for fc in self.shared_fcs:
            bbox_feats = self.relu(fc(bbox_feats))
        cls_score = self.fc_cls(bbox_feats)
        bbox_pred = self.fc_reg(bbox_feats)
        return RCNNOut(cls_score, bbox_pred)

    def __call__(
        self,
        features: List[torch.Tensor],
        boxes: List[torch.Tensor],
    ) -> RCNNOut:
        """Type definition for function call."""
        return self._call_impl(features, boxes)


class TransformRCNNOutputs(nn.Module):
    def __init__(
        self,
        bbox_coder: DeltaXYWHBBoxEncoder,
        score_threshold: float = 0.05,
        iou_threshold: float = 0.5,
        max_per_img: int = 100,
    ) -> None:
        super().__init__()
        self.bbox_coder = bbox_coder
        self.score_threshold = score_threshold
        self.max_per_img = max_per_img
        self.iou_threshold = iou_threshold

    def forward(
        self,
        class_outs: torch.Tensor,
        regression_outs: torch.Tensor,
        boxes: List[torch.Tensor],
        images_shape: Tuple[int, int, int, int],
    ) -> List[Detections]:
        """
        Args:

        Returns:
            Detections
        """
        result_boxes = []

        num_proposals_per_img = tuple(len(p) for p in boxes)
        regression_outs = regression_outs.split(num_proposals_per_img, 0)
        class_outs = class_outs.split(num_proposals_per_img, 0)
        for cls_out, reg_out, boxs in zip(class_outs, regression_outs, boxes):
            scores = F.softmax(cls_out, dim=-1)
            bboxes = self.bbox_coder.decode(
                boxs[:, :4], reg_out, max_shape=images_shape[2:]
            )
            det_bbox, det_scores, det_label = multiclass_nms(
                bboxes,
                scores,
                self.score_threshold,
                self.iou_threshold,
                self.max_per_img,
            )
            result_boxes.append(Detections(det_bbox, det_scores, det_label))

        return result_boxes

    def __call__(
        self,
        class_outs: torch.Tensor,
        regression_outs: torch.Tensor,
        boxes: List[torch.Tensor],
        images_shape: Tuple[int, int, int, int],
    ) -> List[Detections]:
        """Type definition for function call."""
        return self._call_impl(
            class_outs, regression_outs, boxes, images_shape
        )


class RCNNTargets(NamedTuple):
    labels: Tensor
    label_weights: Tensor
    bbox_targets: Tensor
    bbox_weights: Tensor


class RCNNLosses(NamedTuple):
    rcnn_loss_cls: torch.Tensor
    rcnn_loss_bbox: torch.Tensor


class RCNNLoss(nn.Module):
    def __init__(
        self, bbox_coder: DeltaXYWHBBoxEncoder, num_classes: int = 80
    ):
        super().__init__()
        self.num_classes = num_classes
        self.bbox_coder = bbox_coder

    def _get_targets_per_image(
        self,
        pos_bboxes: Tensor,
        neg_bboxes: Tensor,
        pos_gt_bboxes: Tensor,
        pos_gt_labels: Tensor,
    ):
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full(
            (num_samples,), self.num_classes, dtype=torch.long
        )
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            label_weights[:num_pos] = 1.0
            pos_bbox_targets = self.bbox_coder.encode(
                pos_bboxes, pos_gt_bboxes
            )
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0
        return RCNNTargets(labels, label_weights, bbox_targets, bbox_weights)

    def forward(
        self,
        class_outs: torch.Tensor,
        regression_outs: torch.Tensor,
        boxes: List[torch.Tensor],
        boxes_mask,
        target_boxes: List[torch.Tensor],
        target_classes: List[torch.Tensor],
    ):
        """
        M =
        Args:
            class_outs Tensor[M*B, num_classes]
            regression_outs Tensor[M*B, regression_params]
            boxes: List[Tensor[M, 4]] len B
            boxes_mask List[Tensor[M,]] - positive (1), ignore (-1), negative (0)
            target_boxes: List[Tensor[M, 4]]
            target_classes: List[Tensor[M,]]


        """
        # get targets
        targets = []
        for boxs, boxs_mask, tgt_boxs, tgt_cls in zip(
            boxes, boxes_mask, target_boxes, target_classes
        ):
            targets.append(
                self._get_targets_per_image(
                    boxs[boxs_mask == 1],
                    boxs[boxs_mask == 0],
                    tgt_boxs[boxs_mask == 1],
                    tgt_cls[boxs_mask == 1],
                )
            )

        labels = torch.cat([tgt.labels for tgt in targets], 0)
        label_weights = torch.cat([tgt.label_weights for tgt in targets], 0)
        bbox_targets = torch.cat([tgt.bbox_targets for tgt in targets], 0)
        bbox_weights = torch.cat([tgt.bbox_weights for tgt in targets], 0)

        # compute losses
        avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.0)
        if class_outs.numel() > 0:
            loss_cls = weight_reduce_loss(
                F.cross_entropy(class_outs, labels, reduction="none"),
                label_weights,
                avg_factor=avg_factor,
            )
        else:
            loss_cls = class_outs.sum()

        bg_class_ind = self.num_classes
        # 0~self.num_classes-1 are FG, self.num_classes is BG
        pos_inds = (labels >= 0) & (labels < bg_class_ind)
        # do not perform bounding box regression for BG anymore.
        if pos_inds.any():
            pos_reg_outs = regression_outs.view(
                regression_outs.size(0), -1, 4
            )[pos_inds.type(torch.bool), labels[pos_inds.type(torch.bool)]]
            loss_bbox = l1_loss(
                pos_reg_outs,
                bbox_targets[pos_inds.type(torch.bool)],
                bbox_weights[pos_inds.type(torch.bool)],
                avg_factor=bbox_targets.size(0),
            )
        else:
            loss_bbox = regression_outs[pos_inds].sum()

        return RCNNLosses(rcnn_loss_cls=loss_cls, rcnn_loss_bbox=loss_bbox)


class MMDetMaskHead(nn.Module):  # TODO convert into Mask Head implementation
    def forward(self):
        masks = self.mm_roi_head.simple_test_mask(
            feat_list, img_metas, bboxes, labels
        )
        segmentations = segmentations_from_mmdet(
            masks, detections, inputs.device
        )
