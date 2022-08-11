"""Faster RCNN roi head."""
from math import prod
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from vis4d.common.bbox.coders.delta_xywh_coder import DeltaXYWHBBoxCoder
from vis4d.common.bbox.poolers import MultiScaleRoIAlign
from vis4d.common.bbox.utils import multiclass_nms
from vis4d.model.utils import segmentations_from_mmdet
from vis4d.struct import (
    Boxes2D,
    DictStrAny,
    InputSample,
    InstanceMasks,
    LabelInstances,
    LossesType,
    NamedTensors,
)


class FRCNNRoIHeadOutput(NamedTuple):
    cls_score: torch.Tensor
    bbox_pred: torch.Tensor


class Box2DRoIHead(nn.Module):  # TODO rename faster rcnn
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

        # TODO weight init

    def forward(
        self,
        features: List[torch.Tensor],
        boxes: List[torch.Tensor],
    ) -> FRCNNRoIHeadOutput:  # TODO Tobias revisit, do we need list here or can work with tensors
        """Forward pass during training stage."""
        bbox_feats = self.roi_pooler(features, boxes).flatten(start_dim=1)
        for fc in self.shared_fcs:
            bbox_feats = self.relu(fc(bbox_feats))
        cls_score = self.fc_cls(bbox_feats)
        bbox_pred = self.fc_reg(bbox_feats)

        num_proposals_per_img = tuple(len(p) for p in boxes)
        bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        return cls_score, bbox_pred


class TransformMMDetFRCNNRoIHeadOutputs(nn.Module):
    def __init__(
        self,
        score_threshold: float = 0.05,
        iou_threshold: float = 0.5,
        max_per_img: int = 100,
    ) -> None:
        super().__init__()
        self.score_threshold = score_threshold
        self.max_per_img = max_per_img
        self.iou_threshold = iou_threshold
        self.bbox_coder = DeltaXYWHBBoxCoder(
            clip_border=True,
            target_means=(0.0, 0.0, 0.0, 0.0),
            target_stds=(0.1, 0.1, 0.2, 0.2),
        )

    def forward(
        self,
        class_outs: List[torch.Tensor],
        regression_outs: List[torch.Tensor],
        boxes: List[torch.Tensor],
        images_shape: Tuple[int, int, int, int],
    ) -> List[Boxes2D]:
        """
        Args:

        Returns:
            boxes
            scores
            class_ids
        """
        result_boxes = []
        for cls_out, reg_out, boxs in zip(class_outs, regression_outs, boxes):
            scores = F.softmax(cls_out, dim=-1)
            bboxes = self.bbox_coder.decode(
                boxs[:, :4], reg_out, max_shape=images_shape[2:]
            )
            det_bbox, det_label = multiclass_nms(
                bboxes,
                scores,
                self.score_threshold,
                self.iou_threshold,
                self.max_per_img,
            )

            result_boxes.append(Boxes2D(det_bbox, det_label))

        return result_boxes


class MMDetFRCNNRoIHeadLoss(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.mm_roi_head = head

    def forward(
        self,
        class_outs: torch.Tensor,
        regression_outs: [torch.Tensor],
        boxes: List[torch.Tensor],
        boxes_mask,
        target_boxes: List[torch.Tensor],
        target_classes: List[torch.Tensor],
    ):
        """
        M =
        Args:
            class_outs Tensor[M, num_classes]
            regression_outs List[Tensor[M, regression_params]]
            boxes: List[Tensor[M, 4]]
            boxes_mask List[Tensor[M,]] - positive (1), ignore (-1), negative (0)
            target_boxes: List[Tensor[M, 4]]
            target_classes: List[Tensor[M,]]


        """
        pos_bboxes_list = [
            boxs[boxs_mask == 1] for boxs, boxs_mask in zip(boxes, boxes_mask)
        ]
        neg_bboxes_list = [
            boxs[boxs_mask == 0] for boxs, boxs_mask in zip(boxes, boxes_mask)
        ]
        pos_gt_bboxes_list = [
            tgt_boxs[boxs_mask]
            for tgt_boxs, boxs_mask in zip(target_boxes, boxes_mask)
        ]
        pos_gt_labels_list = [
            tgt_cls[boxs_mask == 1]
            for tgt_cls, boxs_mask in zip(target_classes, boxes_mask)
        ]
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            cfg=self.mm_roi_head.test_cfg,
        )

        labels = torch.cat(labels, 0)
        label_weights = torch.cat(label_weights, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        bbox_weights = torch.cat(bbox_weights, 0)

        losses = self.mm_bbox_head.loss(
            class_outs,
            regression_outs,
            boxes,
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
        )
        return losses


class MMDetMaskHead(nn.Module):
    def forward(self):
        masks = self.mm_roi_head.simple_test_mask(
            feat_list, img_metas, bboxes, labels
        )
        segmentations = segmentations_from_mmdet(
            masks, detections, inputs.device
        )
