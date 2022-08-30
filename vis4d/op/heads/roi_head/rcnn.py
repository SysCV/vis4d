"""Faster RCNN roi head."""
from math import prod
from typing import List, NamedTuple, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.ops import roi_align

from vis4d.common.bbox.coders.delta_xywh_coder import DeltaXYWHBBoxEncoder
from vis4d.common.bbox.poolers import MultiScaleRoIAlign
from vis4d.common.bbox.utils import multiclass_nms
from vis4d.common.mask.mask_ops import paste_masks_in_image
from vis4d.op.losses.utils import l1_loss, weight_reduce_loss
from vis4d.op.utils import segmentations_from_mmdet
from vis4d.struct import Detections, Masks


class RCNNOut(NamedTuple):
    """RoI head outputs."""

    # logits for the box classication. The logit dimention is number of classes
    # plus 1 for the background.
    cls_score: torch.Tensor
    # Each box has regression for all classes. So the tensor dimention is
    # [batch_size, number of boxes, number of classes x 4]
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
        """_summary_
        # TODO(tobiasfshr)
        Args:
            num_classes (int, optional): _description_. Defaults to 80.
            roi_size (Tuple[int, int], optional): _description_. Defaults to (7, 7).
            in_channels (int, optional): _description_. Defaults to 256.
            fc_out_channels (int, optional): _description_. Defaults to 1024.
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


class DetOut(NamedTuple):
    """Output of the final detections from RCNN."""

    boxes: List[torch.Tensor]  # N, 4
    scores: List[torch.Tensor]
    class_ids: List[torch.Tensor]
    indices: List[torch.Tensor]


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
        bbox_coder: DeltaXYWHBBoxEncoder,
        score_threshold: float = 0.05,
        iou_threshold: float = 0.5,
        max_per_img: int = 100,
    ) -> None:
        """_summary_

        # TODO(tobiasfshr)

        Args:
            bbox_coder (DeltaXYWHBBoxEncoder): _description_
            score_threshold (float, optional): _description_. Defaults to 0.05.
            iou_threshold (float, optional): _description_. Defaults to 0.5.
            max_per_img (int, optional): _description_. Defaults to 100.
        """
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
    ) -> DetOut:
        """_summary_
        # TODO(tobiasfshr)
        Args:
            class_outs (torch.Tensor): _description_
            regression_outs (torch.Tensor): _description_
            boxes (List[torch.Tensor]): _description_
            images_shape (Tuple[int, int, int, int]): _description_

        Returns:
            List[Detections]: _description_
        """
        num_proposals_per_img = tuple(len(p) for p in boxes)
        regression_outs = regression_outs.split(num_proposals_per_img, 0)
        class_outs = class_outs.split(num_proposals_per_img, 0)
        all_det_boxes = []
        all_det_scores = []
        all_det_class_ids = []
        all_det_inds = []
        for cls_out, reg_out, boxs in zip(class_outs, regression_outs, boxes):
            scores = F.softmax(cls_out, dim=-1)
            bboxes = self.bbox_coder.decode(
                boxs[:, :4], reg_out, max_shape=images_shape[2:]
            )
            det_bbox, det_scores, det_label, indices = multiclass_nms(
                bboxes,
                scores,
                self.score_threshold,
                self.iou_threshold,
                self.max_per_img,
                return_inds=True,
            )
            all_det_boxes.append(det_bbox)
            all_det_scores.append(det_scores)
            all_det_class_ids.append(det_label)
            all_det_inds.append(indices)

        return DetOut(
            boxes=all_det_boxes,
            scores=all_det_scores,
            class_ids=all_det_class_ids,
            indices=all_det_inds,
        )

    def __call__(
        self,
        class_outs: torch.Tensor,
        regression_outs: torch.Tensor,
        boxes: List[torch.Tensor],
        images_shape: Tuple[int, int, int, int],
    ) -> DetOut:
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


class MaskRCNNHead(nn.Module):
    """mask rcnn roi head."""

    def __init__(
        self,
        num_classes: int = 80,
        num_convs: int = 4,
        roi_size: Tuple[int, int] = (14, 14),
        in_channels: int = 256,
        conv_kernel_size: int = 3,
        conv_out_channels: int = 256,
        scale_factor: int = 2,
        class_agnostic: bool = False,
    ) -> None:
        """Init."""
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

    def _init_weights(self, module, mode="fan_in"):
        if hasattr(module, "weight") and hasattr(module, "bias"):
            nn.init.kaiming_normal_(
                module.weight, mode=mode, nonlinearity="relu"
            )
            nn.init.constant_(module.bias, 0)

    def forward(
        self,
        features: List[torch.Tensor],
        boxes: List[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass during training stage."""
        mask_feats = self.roi_pooler(features, boxes)
        for conv in self.convs:
            mask_feats = self.relu(conv(mask_feats))
        mask_feats = self.relu(self.upsample(mask_feats))
        mask_pred = self.conv_logits(mask_feats)
        return mask_pred


class MaskOut(NamedTuple):
    """Output of the final detections from Mask RCNN."""

    masks: List[torch.Tensor]  # N, H, W
    scores: List[torch.Tensor]
    class_ids: List[torch.Tensor]


class Det2Mask(nn.Module):
    """Post processing of mask predictions."""

    def __init__(self, mask_threshold: float = 0.5) -> None:
        """Init.

        Args:
            mask_threshold (float, optional): _description_. Defaults to 0.5.
        """
        super().__init__()
        self.mask_threshold = mask_threshold

    def forward(
        self,
        mask_outs: torch.Tensor,
        dets: DetOut,
        images_shape: Tuple[int, int, int, int],
    ) -> MaskOut:
        """_summary_
        # TODO (thomaseh)

        Args:
            mask_outs (torch.Tensor): _description_
            dets (DetOut): _description_
            images_shape (Tuple[int, int, int, int]): _description_

        Returns:
            MaskOut: _description_
        """
        all_masks = []
        all_scores = []
        all_class_ids = []
        for indices, boxes, scores, class_ids in zip(
            dets.indices, dets.boxes, dets.scores, dets.class_ids
        ):
            pasted_masks = paste_masks_in_image(
                mask_outs[indices // 80, indices % 80],
                boxes,
                images_shape[2:],
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
        mask_outs: torch.Tensor,
        dets: DetOut,
        images_shape: Tuple[int, int, int, int],
    ) -> MaskOut:
        """Type definition for function call."""
        return self._call_impl(mask_outs, dets, images_shape)


class MaskRCNNLosses(NamedTuple):
    rcnn_loss_mask: torch.Tensor


class MaskRCNNLoss(nn.Module):
    """Mask RCNN loss function."""

    def _get_targets_per_image(
        self,
        boxes: Tensor,
        tgt_masks: Tensor,
        out_shape: Tuple[int, int],
        binarize: bool = True,
    ) -> Tensor:
        """_summary_
        # TODO (thomaseh)

        Args:
            boxes (Tensor): _description_
            tgt_masks (Tensor): _description_
            out_shape (Tuple[int, int]): _description_
            binarize (bool, optional): _description_. Defaults to True.

        Returns:
            Tensor: _description_
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
        mask_outs: torch.Tensor,
        proposal_boxes: List[torch.Tensor],
        proposal_labels: List[torch.Tensor],
        target_masks: List[torch.Tensor],
    ) -> MaskRCNNLosses:
        """_summary_
        # TODO (thomaseh)

        Args:
            mask_outs (torch.Tensor): _description_
            proposal_boxes (List[torch.Tensor]): _description_
            proposal_labels (List[torch.Tensor]): _description_
            target_masks (List[torch.Tensor]): _description_

        Returns:
            MaskRCNNLosses: _description_
        """
        mask_size = tuple(mask_outs.shape[2:])
        # get targets
        targets = []
        for boxes, tgt_masks in zip(proposal_boxes, target_masks):
            targets.append(
                self._get_targets_per_image(boxes, tgt_masks, mask_size)
            )
        mask_targets = torch.cat(targets)
        mask_labels = torch.cat(proposal_labels)

        num_rois = mask_outs.shape[0]
        inds = torch.arange(
            0, num_rois, dtype=torch.long, device=mask_outs.device
        )
        pred_slice = mask_outs[inds, mask_labels.long()].squeeze(1)
        loss_mask = F.binary_cross_entropy_with_logits(
            pred_slice, mask_targets.float(), reduction="mean"
        )

        return MaskRCNNLosses(rcnn_loss_mask=loss_mask)
