"""Faster RCNN roi head."""
from __future__ import annotations

from math import prod
from typing import NamedTuple, Protocol

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.ops import roi_align

from vis4d.op.box.box2d import apply_mask, bbox_clip, multiclass_nms
from vis4d.op.box.encoder import BoxEncoder2D
from vis4d.op.box.poolers import MultiScaleRoIAlign
from vis4d.op.layer import add_conv_branch
from vis4d.op.loss.common import l1_loss
from vis4d.op.loss.reducer import SumWeightedLoss
from vis4d.op.mask.util import paste_masks_in_image

from ..typing import Proposals, Targets


class RCNNOut(NamedTuple):
    """RoI head outputs."""

    # logits for the box classication. The logit dimention is number of classes
    # plus 1 for the background.
    cls_score: torch.Tensor
    # Each box has regression for all classes. So the tensor dimention is
    # [batch_size, number of boxes, number of classes x 4]
    bbox_pred: torch.Tensor


class RCNNHead(nn.Module):
    """FasterRCNN RoI head.

    This head pools the RoIs from a set of feature maps and processes them
    into classification / regression outputs.
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
    ) -> None:
        """Creates an instance of the class.

        Args:
            num_classes (int, optional): number of categories. Defaults to 80.
            roi_size (tuple[int, int], optional): size of pooled RoIs. Defaults
                to (7, 7).
            in_channels (int, optional): Number of channels in input feature
                maps. Defaults to 256.
            fc_out_channels (int, optional): Output channels of shared linear
                layers. Defaults to 1024.
        """
        super().__init__()
        self.roi_pooler = MultiScaleRoIAlign(
            sampling_ratio=0, resolution=roi_size, strides=[4, 8, 16, 32]
        )

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

        self._init_weights(self.fc_cls)
        self._init_weights(self.fc_reg, std=0.001)

    def _add_conv_fc_branch(
        self,
        num_branch_convs: int = 0,
        num_branch_fcs: int = 0,
        in_channels: int = 0,
        is_shared: bool = False,
    ) -> tuple[nn.ModuleList, nn.ModuleList, int]:
        """Add shared or separable branch."""
        last_layer_dim = in_channels
        # add branch specific conv layers
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
                last_layer_dim *= np.prod(self.roi_pooler.resolution)
            for i in range(num_branch_fcs):
                fc_in_dim = last_layer_dim if i == 0 else self.fc_out_channels
                fcs.append(nn.Linear(fc_in_dim, self.fc_out_channels))
        return convs, fcs, last_layer_dim

    @staticmethod
    def _init_weights(module: nn.Module, std: float = 0.01) -> None:
        """Init weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self, features: list[torch.Tensor], boxes: list[torch.Tensor]
    ) -> RCNNOut:
        """Forward pass during training stage."""
        # Take stride 4, 8, 16, 32 features
        bbox_feats = self.roi_pooler(features[2:6], boxes)
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                bbox_feats = self.relu(conv(bbox_feats))

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


class DetOut(NamedTuple):
    """Output of the final detections from RCNN."""

    boxes: list[torch.Tensor]  # N, 4
    scores: list[torch.Tensor]
    class_ids: list[torch.Tensor]


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
        box_encoder: BoxEncoder2D,
        score_threshold: float = 0.05,
        iou_threshold: float = 0.5,
        max_per_img: int = 100,
    ) -> None:
        """Creates an instance of the class.

        Args:
            box_encoder (BoxEncoder2D): Decodes regression parameters to
                detected boxes.
            score_threshold (float, optional): Minimum score of a detection.
                Defaults to 0.05.
            iou_threshold (float, optional): IoU threshold of NMS
                post-processing step. Defaults to 0.5.
            max_per_img (int, optional): Maximum number of detections per
                image. Defaults to 100.
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
                self.bbox_coder.decode(boxs[:, :4], reg_out).view(-1, 4),
                image_hw,
            ).view(reg_out.shape)
            det_bbox, det_scores, det_label, _ = multiclass_nms(
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
    """RCNN loss in FasterRCNN.

    This class computes the loss of RCNN given proposal boxes and their
    corresponding target boxes with the given box encoder.
    """

    def __init__(
        self, box_encoder: BoxEncoder2D, num_classes: int = 80
    ) -> None:
        """Creates an instance of the class.

        Args:
            box_encoder (BoxEncoder2D): Decodes box regression parameters into
                detected boxes.
            num_classes (int, optional): number of object categories. Defaults
                to 80.
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
            boxes (Tensor): [N, 4] tensor of proposal boxes
            labels (Tensor): [N,] tensor of positive / negative / ignore labels
            target_boxes (Tensor): [N, 4] Assigned target boxes.
            target_classes (Tensor): [N,] Assigned target class labels.

        Returns:
            RCNNTargets: Box / class label tensors and weights.
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
        avg_factor = torch.sum(label_weights > 0).clamp(1.0)
        if class_outs.numel() > 0:
            loss_cls = SumWeightedLoss(label_weights, avg_factor)(
                F.cross_entropy(class_outs, labels, reduction="none")
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
                SumWeightedLoss(
                    bbox_weights[pos_inds.type(torch.bool)],
                    bbox_targets.size(0),
                ),
            )
        else:
            loss_bbox = regression_outs[pos_inds].sum()

        return RCNNLosses(rcnn_loss_cls=loss_cls, rcnn_loss_bbox=loss_bbox)


class MaskRCNNHeadOut(NamedTuple):
    """Mask RCNN RoI head outputs."""

    # logits for mask prediction. The dimension is number of masks x number of
    # classes x H_mask x W_mask
    mask_pred: list[torch.Tensor]


class MaskRCNNHead(nn.Module):
    """mask rcnn roi head."""

    def __init__(
        self,
        num_classes: int = 80,
        num_convs: int = 4,
        roi_size: tuple[int, int] = (14, 14),
        in_channels: int = 256,
        conv_kernel_size: int = 3,
        conv_out_channels: int = 256,
        scale_factor: int = 2,
        class_agnostic: bool = False,
    ) -> None:
        """Creates an instance of the class.

        Args:
            num_classes (int, optional): Number of classes. Defaults to 80.
            num_convs (int, optional): Number of convolution layers. Defaults
                to 4.
            roi_size (tuple[int, int], optional): Size of RoI after pooling.
                Defaults to (14, 14).
            in_channels (int, optional): Input feature channels.
                Defaults to 256.
            conv_kernel_size (int, optional): Kernel size of convolution.
                Defaults to 3.
            conv_out_channels (int, optional): Output channels of convolution.
                Defaults to 256.
            scale_factor (int, optional): Scaling factor of upsampling.
                Defaults to 2.
            class_agnostic (bool, optional): Whether to do class agnostic mask
                prediction. Defaults to False.
        """
        super().__init__()
        self.roi_pooler = MultiScaleRoIAlign(
            sampling_ratio=0, resolution=roi_size, strides=[4, 8, 16, 32]
        )

        self.convs = nn.ModuleList()
        for i in range(num_convs):
            in_channels = in_channels if i == 0 else conv_out_channels
            padding = (conv_kernel_size - 1) // 2
            self.convs.append(
                nn.Conv2d(
                    in_channels,
                    conv_out_channels,
                    conv_kernel_size,
                    padding=padding,
                )
            )

        upsample_in_channels = (
            conv_out_channels if num_convs > 0 else in_channels
        )
        self.upsample = nn.ConvTranspose2d(
            upsample_in_channels,
            conv_out_channels,
            scale_factor,
            stride=scale_factor,
        )

        out_channels = 1 if class_agnostic else num_classes
        self.conv_logits = nn.Conv2d(conv_out_channels, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)

        self._init_weights(self.convs)
        self._init_weights(self.upsample, mode="fan_out")
        self._init_weights(self.conv_logits, mode="fan_out")

    @staticmethod
    def _init_weights(module: nn.Module, mode: str = "fan_in") -> None:
        """Initialize weights."""
        if hasattr(module, "weight") and hasattr(module, "bias"):
            assert isinstance(module.weight, torch.Tensor) and isinstance(
                module.bias, torch.Tensor
            )
            nn.init.kaiming_normal_(
                module.weight, mode=mode, nonlinearity="relu"
            )
            nn.init.constant_(module.bias, 0)

    def forward(
        self, features: list[torch.Tensor], boxes: list[torch.Tensor]
    ) -> MaskRCNNHeadOut:
        """Forward pass.

        Args:
            features (list[torch.Tensor]): Feature pyramid.
            boxes (list[torch.Tensor]): Proposal boxes.

        Returns:
            MaskRCNNHeadOut: Mask prediction outputs.
        """
        # Take stride 4, 8, 16, 32 features
        mask_feats = self.roi_pooler(features[2:6], boxes)
        for conv in self.convs:
            mask_feats = self.relu(conv(mask_feats))
        mask_feats = self.relu(self.upsample(mask_feats))
        mask_pred = self.conv_logits(mask_feats)
        num_dets_per_img = tuple(len(d) for d in boxes)
        mask_preds = mask_pred.split(num_dets_per_img, 0)
        return MaskRCNNHeadOut(mask_pred=mask_preds)


class MaskOut(NamedTuple):
    """Output of the final detections from Mask RCNN."""

    masks: list[torch.Tensor]  # N, H, W
    scores: list[torch.Tensor]
    class_ids: list[torch.Tensor]


class Det2Mask(nn.Module):
    """Post processing of mask predictions."""

    def __init__(self, mask_threshold: float = 0.5) -> None:
        """Creates an instance of the class.

        Args:
            mask_threshold (float, optional): Positive threshold. Defaults to
                0.5.
        """
        super().__init__()
        self.mask_threshold = mask_threshold

    def forward(
        self,
        mask_outs: list[torch.Tensor],
        det_boxes: list[torch.Tensor],
        det_scores: list[torch.Tensor],
        det_class_ids: list[torch.Tensor],
        original_hw: list[tuple[int, int]],
    ) -> MaskOut:
        """Paste mask predictions back into original image resolution.

        Args:
            mask_outs (list[torch.Tensor]): List of mask outputs for each batch
                element.
            det_boxes (list[torch.Tensor]): List of detection boxes for each
                batch element.
            det_scores (list[torch.Tensor]): List of detection scores for each
                batch element.
            det_class_ids (list[torch.Tensor]): List of detection classeds for
                each batch element.
            original_hw (list[tuple[int, int]]): Original image resolution.

        Returns:
            MaskOut: Post-processed mask predictions.
        """
        all_masks = []
        all_scores = []
        all_class_ids = []
        for mask_out, boxes, scores, class_ids, orig_hw in zip(
            mask_outs, det_boxes, det_scores, det_class_ids, original_hw
        ):
            pasted_masks = paste_masks_in_image(
                mask_out[torch.arange(len(mask_out)), class_ids],
                boxes,
                orig_hw[::-1],
                self.mask_threshold,
            )
            all_masks.append(pasted_masks)
            all_scores.append(scores)
            all_class_ids.append(class_ids)
        return MaskOut(
            masks=all_masks, scores=all_scores, class_ids=all_class_ids
        )

    def __call__(
        self,
        mask_outs: list[torch.Tensor],
        det_boxes: list[torch.Tensor],
        det_scores: list[torch.Tensor],
        det_class_ids: list[torch.Tensor],
        original_hw: list[tuple[int, int]],
    ) -> MaskOut:
        """Type definition for function call."""
        return self._call_impl(
            mask_outs, det_boxes, det_scores, det_class_ids, original_hw
        )


class MaskRCNNHeadLosses(NamedTuple):
    """Mask RoI head loss container."""

    rcnn_loss_mask: torch.Tensor


# TODO, why is this here and not in losses.py?
class MaskRCNNHeadLoss(nn.Module):
    """Mask RoI head loss function."""

    def __init__(self, num_classes: int = 80) -> None:
        """Creates an instance of the class.

        Args:
            num_classes (int, optional): number of object categories. Defaults
                to 80.
        """
        super().__init__()
        self.num_classes = num_classes

    @staticmethod
    def _get_targets_per_image(
        boxes: Tensor,
        tgt_masks: Tensor,
        out_shape: tuple[int, int],
        binarize: bool = True,
    ) -> Tensor:
        """Get aligned mask targets for each proposal.

        Args:
            boxes (Tensor): proposal boxes.
            tgt_masks (Tensor): target masks.
            out_shape (tuple[int, int]): output shape.
            binarize (bool, optional): whether to convert target mask to
                binary. Defaults to True.

        Returns:
            Tensor: aligned mask targets.
        """
        fake_inds = torch.arange(len(boxes), device=boxes.device)[:, None]
        rois = torch.cat([fake_inds, boxes], dim=1)  # Nx5
        gt_masks_th = tgt_masks[:, None, :, :].type(rois.dtype)
        targets = roi_align(
            gt_masks_th, rois, out_shape, 1.0, 0, True
        ).squeeze(1)
        resized_masks = targets >= 0.5 if binarize else targets
        return resized_masks

    def forward(
        self,
        mask_preds: list[torch.Tensor],
        proposal_boxes: list[torch.Tensor],
        target_classes: list[torch.Tensor],
        target_masks: list[torch.Tensor],
    ) -> MaskRCNNHeadLosses:
        """Calculate losses of Mask RCNN head.

        Args:
            mask_preds (list[torch.Tensor]): [M, C, H', W'] mask outputs per
                batch element.
            proposal_boxes (list[torch.Tensor]): [M, 4] proposal boxes per
                batch element.
            target_classes (list[torch.Tensor]): list of [M, 4] assigned
                target boxes for each proposal.
            target_masks (list[torch.Tensor]): list of [M, H, W] assigned
                target masks for each proposal.

        Returns:
            MaskRCNNHeadLosses: mask loss.
        """
        mask_pred = torch.cat(mask_preds)
        mask_size = (mask_pred.shape[2], mask_pred.shape[3])
        # get targets
        targets = []
        for boxes, tgt_masks in zip(proposal_boxes, target_masks):
            if len(tgt_masks) == 0:
                targets.append(
                    torch.empty((0, *mask_size), device=tgt_masks.device)
                )
            else:
                targets.append(
                    self._get_targets_per_image(boxes, tgt_masks, mask_size)
                )
        mask_targets = torch.cat(targets)
        mask_labels = torch.cat(target_classes)

        if len(mask_targets) > 0:
            num_rois = mask_pred.shape[0]
            inds = torch.arange(
                0, num_rois, dtype=torch.long, device=mask_pred.device
            )
            pred_slice = mask_pred[inds, mask_labels[inds].long()].squeeze(1)
            loss_mask = F.binary_cross_entropy_with_logits(
                pred_slice, mask_targets.float(), reduction="mean"
            )
        else:
            loss_mask = mask_targets.sum()

        return MaskRCNNHeadLosses(rcnn_loss_mask=loss_mask)


class MaskSampler(Protocol):
    """Type definition for mask sampler."""

    def __call__(
        self,
        target_masks: list[Tensor],
        sampled_target_indices: list[Tensor],
        sampled_targets: Targets,
        sampled_proposals: Proposals,
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        """Type definition for function call.

        Args:
            target_masks (list[Tensor]): list of [N, H, W] target masks per
                batch element.
            sampled_target_indices (list[Tensor]): list of [M] indices of
                sampled targets per batch element.
            sampled_targets (Targets): sampled targets.
            sampled_proposals (Proposals): sampled proposals.

        Returns:
            tuple[list[Tensor], list[Tensor], list[Tensor]]: sampled masks,
                sampled target indices, sampled targets.
        """


def positive_mask_sampler(  # TODO: Maybe move to op?
    target_masks: list[Tensor],
    sampled_target_indices: list[Tensor],
    sampled_targets: Targets,
    sampled_proposals: Proposals,
) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
    """Sample only positive masks from target masks.

    Args:
        target_masks (list[Tensor]): list of [N, H, W] target masks per
            batch element.
        sampled_target_indices (list[Tensor]): list of [M] indices of
            sampled targets per batch element.
        sampled_targets (Targets): sampled targets.
        sampled_proposals (Proposals): sampled proposals.

    Returns:
        tuple[list[Tensor], list[Tensor], list[Tensor]]: sampled masks,
            sampled target indices, sampled targets.
    """
    sampled_masks = apply_mask(sampled_target_indices, target_masks)[0]

    pos_proposals, pos_classes, pos_mask_targets = apply_mask(
        [label == 1 for label in sampled_targets.labels],
        sampled_proposals.boxes,
        sampled_targets.classes,
        sampled_masks,
    )
    return pos_proposals, pos_classes, pos_mask_targets


class SampledMaskLoss(nn.Module):
    """Sampled Mask RCNN head loss function."""

    def __init__(
        self,
        mask_sampler: MaskSampler,
        loss: MaskRCNNHeadLoss,
    ) -> None:
        """Initialize sampled mask loss.

        Args:
            mask_sampler (MaskSampler): mask sampler.
            loss (MaskRCNNHeadLoss): mask loss.
        """
        super().__init__()
        self.loss = loss
        self.mask_sampler = mask_sampler

    def forward(
        self,
        mask_preds: list[Tensor],
        target_masks: list[Tensor],
        sampled_target_indices: list[Tensor],
        sampled_targets: Targets,
        sampled_proposals: Proposals,
    ) -> MaskRCNNHeadLosses:
        """Calculate losses of Mask RCNN head.

        Args:
            mask_preds (list[torch.Tensor]): [M, C, H', W'] mask outputs per
                batch element.
            target_masks (list[torch.Tensor]): list of [M, H, W] assigned
                target masks for each proposal.
            sampled_target_indices (list[Tensor]): list of [M, 4] assigned
                target boxes for each proposal.
            sampled_targets (Targets): list of [M, 4] assigned
                target boxes for each proposal.
            sampled_proposals (Proposals): list of [M, 4] assigned
                target boxes for each proposal.

        Returns:
            MaskRCNNHeadLosses: mask loss.
        """
        pos_proposals, pos_classes, pos_mask_targets = self.mask_sampler(
            target_masks,
            sampled_target_indices,
            sampled_targets,
            sampled_proposals,
        )
        return self.loss(
            mask_preds, pos_proposals, pos_classes, pos_mask_targets
        )
