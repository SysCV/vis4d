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


class RCNNOut(NamedTuple):
    """RoI head outputs."""

    # logits for the box classication. The logit dimention is number of classes
    # plus 1 for the background.
    cls_score: torch.Tensor
    # Each box has regression for all classes. So the tensor dimention is
    # [batch_size, number of boxes, number of classes x 4]
    bbox_pred: torch.Tensor


class RCNNHead(nn.Module):
    """FasterRCNN RoI head."""

    def __init__(
        self,
        num_classes: int = 80,
        roi_size: Tuple[int, int] = (7, 7),
        in_channels: int = 256,
        fc_out_channels: int = 1024,
    ) -> None:
        """Init.

        Args:
            num_classes (int, optional): number of categories. Defaults to 80.
            roi_size (Tuple[int, int], optional): size of pooled RoIs. Defaults to (7, 7).
            in_channels (int, optional): Number of channels in input feature maps. Defaults to 256.
            fc_out_channels (int, optional): Output channels of shared linear layers. Defaults to 1024.
        """
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

    def _init_weights(self, module, std: float = 0.01):
        """Init weights."""
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
        # Take stride 4, 8, 16, 32 features
        bbox_feats = self.roi_pooler(features[2:6], boxes).flatten(start_dim=1)
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


class DetOut(NamedTuple):
    """Output of the final detections from RCNN."""

    boxes: List[torch.Tensor]  # N, 4
    scores: List[torch.Tensor]
    class_ids: List[torch.Tensor]


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
        box_encoder: DeltaXYWHBBoxEncoder,
        score_threshold: float = 0.05,
        iou_threshold: float = 0.5,
        max_per_img: int = 100,
    ) -> None:
        """Init.

        Args:
            box_encoder (DeltaXYWHBBoxEncoder): Decodes regression parameters to detected boxes.
            score_threshold (float, optional): Minimum score of a detection. Defaults to 0.05.
            iou_threshold (float, optional): IoU threshold of NMS post-processing step. Defaults to 0.5.
            max_per_img (int, optional): Maximum number of detections per image. Defaults to 100.
        """
        super().__init__()
        self.bbox_coder = box_encoder
        self.score_threshold = score_threshold
        self.max_per_img = max_per_img
        self.iou_threshold = iou_threshold

    def forward(
        self,
        class_outs: torch.Tensor,
        regression_outs: torch.Tensor,
        boxes: List[torch.Tensor],
        images_hw: List[Tuple[int, int]],
    ) -> DetOut:
        """Convert RCNN network outputs to detections.

        Args:
            class_outs (torch.Tensor): [B, N, num_classes] batched tensor of classifiation scores.
            regression_outs (torch.Tensor): [B, N, num_classes * 4] predicted box offsets.
            boxes (List[torch.Tensor]): Initial boxes (RoIs).
            images_hw (List[Tuple[int, int]]): Image sizes.

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
            bboxes = self.bbox_coder.decode(
                boxs[:, :4], reg_out, max_shape=image_hw
            )
            det_bbox, det_scores, det_label = multiclass_nms(
                bboxes,
                scores,
                self.score_threshold,
                self.iou_threshold,
                self.max_per_img,
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
        boxes: List[torch.Tensor],
        images_hw: List[Tuple[int, int]],
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
    """RCNN loss in FasterRCNN."""

    def __init__(
        self, box_encoder: DeltaXYWHBBoxEncoder, num_classes: int = 80
    ):
        """Init.

        Args:
            box_encoder (DeltaXYWHBBoxEncoder): Decodes box regression parameters into detected boxes.
            num_classes (int, optional): number of object categories. Defaults to 80.
        """
        super().__init__()
        self.num_classes = num_classes
        self.box_encoder = box_encoder

    def _get_targets_per_image(
        self,
        boxes: Tensor,
        labels: Tensor,
        target_boxes: Tensor,
        target_classes: Tensor,
    ) -> RCNNTargets:
        """Generate targets per image.

        Args:
            boxes (Tensor): _description_
            labels (Tensor): _description_
            target_boxes (Tensor): _description_
            target_classes (Tensor): _description_

        Returns:
            RCNNTargets: _description_
        """
        pos_mask, neg_mask = labels == 1, labels == 0
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
            pos_box_targets = self.box_encoder.encode(
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
        boxes: List[torch.Tensor],
        boxes_mask: List[torch.Tensor],
        target_boxes: List[torch.Tensor],
        target_classes: List[torch.Tensor],
    ) -> RCNNLosses:
        """Calculate losses of RCNN head.

        Args:
            class_outs (torch.Tensor): [M*B, num_classes]
            regression_outs (torch.Tensor): Tensor[M*B, regression_params]
            boxes (List[torch.Tensor]): [M, 4] len B
            boxes_mask (List[torch.Tensor]): positive (1), ignore (-1), negative (0)
            target_boxes (List[torch.Tensor]): [M, 4] len B
            target_classes (List[torch.Tensor]): [M,] len B

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
