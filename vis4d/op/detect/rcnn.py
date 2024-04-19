"""Faster R-CNN RoI head."""

from __future__ import annotations

from math import prod
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from vis4d.common.typing import TorchLossFunc
from vis4d.op.box.box2d import bbox_clip, multiclass_nms
from vis4d.op.box.encoder import DeltaXYWHBBoxDecoder, DeltaXYWHBBoxEncoder
from vis4d.op.box.poolers import MultiScaleRoIAlign
from vis4d.op.detect.common import DetOut
from vis4d.op.layer import add_conv_branch
from vis4d.op.layer.weight_init import kaiming_init, normal_init, xavier_init
from vis4d.op.loss.common import l1_loss
from vis4d.op.loss.reducer import SumWeightedLoss


class RCNNOut(NamedTuple):
    """Faster R-CNN RoI head outputs."""

    # Logits for box classication. The logit dimension is number of classes
    # plus 1 for the background.
    cls_score: torch.Tensor
    # Each box has regression for all classes. So the tensor dimention is
    # [batch_size, number of boxes, number of classes x 4]
    bbox_pred: torch.Tensor


def get_default_rcnn_box_codec(
    target_means: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    target_stds: tuple[float, float, float, float] = (0.1, 0.1, 0.2, 0.2),
) -> tuple[DeltaXYWHBBoxEncoder, DeltaXYWHBBoxDecoder]:
    """Get the default bounding box encoder and decoder for RCNN."""
    return (
        DeltaXYWHBBoxEncoder(target_means, target_stds),
        DeltaXYWHBBoxDecoder(target_means, target_stds),
    )


class RCNNHead(nn.Module):
    """Faster R-CNN RoI head.

    This head pools the RoIs from a set of feature maps and processes them
    into classification / regression outputs.

    Args:
        num_shared_convs (int, optional): number of shared conv layers.
            Defaults to 0.
        num_shared_fcs (int, optional): number of shared fc layers. Defaults
            to 2.
        conv_out_channels (int, optional): number of output channels for
            shared conv layers. Defaults to 256.
        in_channels (int, optional): Number of channels in input feature maps.
            Defaults to 256.
        fc_out_channels (int, optional): Output channels of shared linear
            layers. Defaults to 1024.
        num_classes (int, optional): number of categories. Defaults to 80.
        roi_size (tuple[int, int], optional): size of pooled RoIs. Defaults
            to (7, 7).
    """

    def __init__(
        self,
        num_shared_convs: int = 0,
        num_shared_fcs: int = 2,
        conv_out_channels: int = 256,
        in_channels: int = 256,
        fc_out_channels: int = 1024,
        num_classes: int = 80,
        roi_size: tuple[int, int] = (7, 7),
        start_level: int = 2,
    ) -> None:
        """Creates an instance of the class."""
        super().__init__()
        self.roi_pooler = MultiScaleRoIAlign(
            sampling_ratio=0, resolution=roi_size, strides=[4, 8, 16, 32]
        )

        # Used feature layers are [start_level, end_level)
        self.start_level = start_level
        self.end_level = start_level + len(self.roi_pooler.scales)

        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels

        # add shared convs and fcs
        (
            self.shared_convs,
            self.shared_fcs,
            last_layer_dim,
        ) = self._add_conv_fc_branch(
            self.num_shared_convs, self.num_shared_fcs, in_channels, True
        )
        self.shared_out_channels = last_layer_dim

        in_channels *= prod(roi_size)

        self.fc_cls = nn.Linear(
            in_features=fc_out_channels, out_features=num_classes + 1
        )
        self.fc_reg = nn.Linear(
            in_features=fc_out_channels, out_features=4 * num_classes
        )
        self.relu = nn.ReLU(inplace=True)

        self._init_weights()

    def _add_conv_fc_branch(
        self,
        num_branch_convs: int = 0,
        num_branch_fcs: int = 0,
        in_channels: int = 0,
        is_shared: bool = False,
    ) -> tuple[nn.ModuleList, nn.ModuleList, int]:
        """Add shared or separable branch."""
        convs, last_layer_dim = add_conv_branch(
            num_branch_convs,
            in_channels,
            self.conv_out_channels,
            True,
            None,
            None,
        )

        fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            if is_shared or num_branch_fcs == 0:
                last_layer_dim *= int(np.prod(self.roi_pooler.resolution))
            for i in range(num_branch_fcs):
                fc_in_dim = last_layer_dim if i == 0 else self.fc_out_channels
                fcs.append(nn.Linear(fc_in_dim, self.fc_out_channels))
        return convs, fcs, last_layer_dim

    def _init_weights(self) -> None:
        """Init weights."""
        for m in self.shared_convs.modules():
            kaiming_init(m)

        for m in self.shared_fcs.modules():
            xavier_init(m, distribution="uniform")

        normal_init(self.fc_cls, std=0.01)
        normal_init(self.fc_reg, std=0.001)

    def forward(
        self, features: list[torch.Tensor], boxes: list[torch.Tensor]
    ) -> RCNNOut:
        """Forward pass during training stage."""
        bbox_feats = self.roi_pooler(
            features[self.start_level : self.end_level], boxes
        )
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                bbox_feats = conv(bbox_feats)

        bbox_feats = bbox_feats.flatten(start_dim=1)

        for fc in self.shared_fcs:
            bbox_feats = self.relu(fc(bbox_feats))
        cls_score = self.fc_cls(bbox_feats)
        bbox_pred = self.fc_reg(bbox_feats)
        return RCNNOut(cls_score, bbox_pred)

    def __call__(
        self, features: list[torch.Tensor], boxes: list[torch.Tensor]
    ) -> RCNNOut:
        """Type definition for function call."""
        return self._call_impl(features, boxes)


class RoI2Det(nn.Module):
    """Post processing of RCNN results and detection generation.

    It does the following:
    1. Take the classification and regression outputs from the RCNN heads.
    2. Take the proposal boxes that are RCNN inputs.
    3. Determine the final box classes and take the according box regression
       parameters.
    4. Adjust the box sizes and offsets according the regression parameters.
    5. Return the final boxes.
    """

    def __init__(
        self,
        box_decoder: None | DeltaXYWHBBoxDecoder = None,
        score_threshold: float = 0.05,
        iou_threshold: float = 0.5,
        max_per_img: int = 100,
        class_agnostic_nms: bool = False,
    ) -> None:
        """Creates an instance of the class.

        Args:
            box_decoder (DeltaXYWHBBoxDecoder, optional): Decodes regression
                parameters to detected boxes. Defaults to None. If None, it
                will use the default decoder.
            score_threshold (float, optional): Minimum score of a detection.
                Defaults to 0.05.
            iou_threshold (float, optional): IoU threshold of NMS
                post-processing step. Defaults to 0.5.
            max_per_img (int, optional): Maximum number of detections per
                image. Defaults to 100.
            class_agnostic_nms (bool, optional): Whether to use class agnostic
                NMS. Defaults to False.
        """
        super().__init__()
        if box_decoder is None:
            _, self.box_decoder = get_default_rcnn_box_codec()
        else:
            self.box_decoder = box_decoder
        self.score_threshold = score_threshold
        self.max_per_img = max_per_img
        self.iou_threshold = iou_threshold
        self.class_agnostic_nms = class_agnostic_nms

    def forward(
        self,
        class_outs: torch.Tensor,
        regression_outs: torch.Tensor,
        boxes: list[torch.Tensor],
        images_hw: list[tuple[int, int]],
    ) -> DetOut:
        """Convert RCNN network outputs to detections.

        Args:
            class_outs (torch.Tensor): [B, num_classes] batched tensor of
                classifiation scores.
            regression_outs (torch.Tensor): [B, num_classes * 4] predicted
                box offsets.
            boxes (list[torch.Tensor]): Initial boxes (RoIs).
            images_hw (list[tuple[int, int]]): Image sizes.

        Returns:
            DetOut: boxes, scores and class ids of detections per image.
        """
        num_proposals_per_img = tuple(len(p) for p in boxes)
        regression_outs = regression_outs.split(num_proposals_per_img, 0)
        class_outs = class_outs.split(num_proposals_per_img, 0)
        all_det_boxes = []
        all_det_scores = []
        all_det_class_ids = []
        for cls_out, reg_out, boxs, image_hw in zip(
            class_outs, regression_outs, boxes, images_hw
        ):
            scores = F.softmax(cls_out, dim=-1)
            bboxes = bbox_clip(
                self.box_decoder(boxs[:, :4], reg_out).view(-1, 4),
                image_hw,
            ).view(reg_out.shape)
            det_bbox, det_scores, det_label, _ = multiclass_nms(
                bboxes,
                scores,
                self.score_threshold,
                self.iou_threshold,
                self.max_per_img,
                self.class_agnostic_nms,
            )
            all_det_boxes.append(det_bbox)
            all_det_scores.append(det_scores)
            all_det_class_ids.append(det_label)

        return DetOut(
            boxes=all_det_boxes,
            scores=all_det_scores,
            class_ids=all_det_class_ids,
        )

    def __call__(
        self,
        class_outs: torch.Tensor,
        regression_outs: torch.Tensor,
        boxes: list[torch.Tensor],
        images_hw: list[tuple[int, int]],
    ) -> DetOut:
        """Type definition for function call."""
        return self._call_impl(class_outs, regression_outs, boxes, images_hw)


class RCNNTargets(NamedTuple):
    """Target container."""

    labels: Tensor
    label_weights: Tensor
    bbox_targets: Tensor
    bbox_weights: Tensor


class RCNNLosses(NamedTuple):
    """RCNN loss container."""

    rcnn_loss_cls: torch.Tensor
    rcnn_loss_bbox: torch.Tensor


class RCNNLoss(nn.Module):
    """RCNN loss in Faster R-CNN.

    This class computes the loss of RCNN given proposal boxes and their
    corresponding target boxes with the given box encoder.
    """

    def __init__(
        self,
        box_encoder: DeltaXYWHBBoxEncoder,
        num_classes: int = 80,
        loss_cls: TorchLossFunc = F.cross_entropy,
        loss_bbox: TorchLossFunc = l1_loss,
    ) -> None:
        """Creates an instance of the class.

        Args:
            box_encoder (DeltaXYWHBBoxEncoder): Decodes box regression
                parameters into detected boxes.
            num_classes (int, optional): number of object categories. Defaults
                to 80.
            loss_cls (TorchLossFunc, optional): Classification loss function.
                Defaults to F.cross_entropy.
            loss_bbox (TorchLossFunc, optional): Regression loss function.
                Defaults to l1_loss.
        """
        super().__init__()
        self.num_classes = num_classes
        self.box_encoder = box_encoder
        self.loss_cls = loss_cls
        self.loss_bbox = loss_bbox

    def _get_targets_per_image(
        self,
        boxes: Tensor,
        labels: Tensor,
        target_boxes: Tensor,
        target_classes: Tensor,
    ) -> RCNNTargets:
        """Generate targets per image.

        Args:
            boxes (Tensor): [N, 4] tensor of proposal boxes
            labels (Tensor): [N,] tensor of positive / negative / ignore labels
            target_boxes (Tensor): [N, 4] Assigned target boxes.
            target_classes (Tensor): [N,] Assigned target class labels.

        Returns:
            RCNNTargets: Box / class label tensors and weights.
        """
        pos_mask, neg_mask = torch.eq(labels, 1), torch.eq(labels, 0)
        num_pos, num_neg = int(pos_mask.sum()), int(neg_mask.sum())
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = boxes.new_full(
            (num_samples,), self.num_classes, dtype=torch.long
        )
        label_weights = boxes.new_zeros(num_samples)
        box_targets = boxes.new_zeros(num_samples, 4)
        box_weights = boxes.new_zeros(num_samples, 4)
        if num_pos > 0:
            pos_target_boxes = target_boxes[pos_mask]
            pos_target_classes = target_classes[pos_mask]
            labels[:num_pos] = pos_target_classes
            label_weights[:num_pos] = 1.0
            pos_box_targets = self.box_encoder(
                boxes[pos_mask], pos_target_boxes
            )
            box_targets[:num_pos, :] = pos_box_targets
            box_weights[:num_pos, :] = 1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0
        return RCNNTargets(labels, label_weights, box_targets, box_weights)

    def forward(
        self,
        class_outs: torch.Tensor,
        regression_outs: torch.Tensor,
        boxes: list[torch.Tensor],
        boxes_mask: list[torch.Tensor],
        target_boxes: list[torch.Tensor],
        target_classes: list[torch.Tensor],
    ) -> RCNNLosses:
        """Calculate losses of RCNN head.

        Args:
            class_outs (torch.Tensor): [M*B, num_classes] classification
                outputs.
            regression_outs (torch.Tensor): Tensor[M*B, regression_params]
                regression outputs.
            boxes (list[torch.Tensor]): [M, 4] proposal boxes per batch
                element.
            boxes_mask (list[torch.Tensor]): positive (1), ignore (-1),
                negative (0).
            target_boxes (list[torch.Tensor]): list of [M, 4] assigned target
                boxes for each proposal.
            target_classes (list[torch.Tensor]): list of [M,] assigned target
                classes for each proposal.

        Returns:
            RCNNLosses: classification and regression losses.
        """
        # get targets
        targets = []
        for boxs, boxs_mask, tgt_boxs, tgt_cls in zip(
            boxes, boxes_mask, target_boxes, target_classes
        ):
            targets.append(
                self._get_targets_per_image(boxs, boxs_mask, tgt_boxs, tgt_cls)
            )

        labels = torch.cat([tgt.labels for tgt in targets], 0)
        label_weights = torch.cat([tgt.label_weights for tgt in targets], 0)
        bbox_targets = torch.cat([tgt.bbox_targets for tgt in targets], 0)
        bbox_weights = torch.cat([tgt.bbox_weights for tgt in targets], 0)

        # compute losses
        avg_factor = torch.sum(torch.greater(label_weights, 0)).clamp(1.0)
        if class_outs.numel() > 0:
            loss_cls = SumWeightedLoss(label_weights, avg_factor)(
                self.loss_cls(class_outs, labels, reduction="none")
            )
        else:
            loss_cls = class_outs.sum()

        bg_class_ind = self.num_classes
        # 0~self.num_classes-1 are FG, self.num_classes is BG
        pos_inds = torch.logical_and(
            torch.greater_equal(labels, 0), torch.less(labels, bg_class_ind)
        )
        # do not perform bounding box regression for BG anymore.
        if pos_inds.any():
            pos_reg_outs = regression_outs.view(
                regression_outs.size(0), -1, 4
            )[pos_inds.type(torch.bool), labels[pos_inds.type(torch.bool)]]
            loss_bbox = self.loss_bbox(
                pred=pos_reg_outs,
                target=bbox_targets[pos_inds.type(torch.bool)],
                reducer=SumWeightedLoss(
                    bbox_weights[pos_inds.type(torch.bool)],
                    bbox_targets.size(0),
                ),
            )
        else:
            loss_bbox = regression_outs[pos_inds].sum()

        return RCNNLosses(rcnn_loss_cls=loss_cls, rcnn_loss_bbox=loss_bbox)
