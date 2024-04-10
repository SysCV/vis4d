"""Mask RCNN detector."""

from __future__ import annotations

from typing import NamedTuple, Protocol

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.ops import roi_align

from vis4d.op.box.box2d import apply_mask
from vis4d.op.box.poolers import MultiScaleRoIAlign
from vis4d.op.mask.util import paste_masks_in_image, remove_overlap

from ..typing import Proposals, Targets


class MaskRCNNHeadOut(NamedTuple):
    """Mask R-CNN RoI head outputs."""

    # logits for mask prediction. The dimension is number of masks x number of
    # classes x H_mask x W_mask
    mask_pred: list[torch.Tensor]


class MaskRCNNHead(nn.Module):
    """Mask R-CNN RoI head.

    Args:
        num_classes (int, optional): Number of classes. Defaults to 80.
        num_convs (int, optional): Number of convolution layers. Defaults to 4.
        roi_size (tuple[int, int], optional): Size of RoI after pooling.
            Defaults to (14, 14).
        in_channels (int, optional): Input feature channels. Defaults to 256.
        conv_kernel_size (int, optional): Kernel size of convolution. Defaults
            to 3.
        conv_out_channels (int, optional): Output channels of convolution.
            Defaults to 256.
        scale_factor (int, optional): Scaling factor of upsampling. Defaults
            to 2.
        class_agnostic (bool, optional): Whether to do class agnostic mask
            prediction. Defaults to False.
    """

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
        """Creates an instance of the class."""
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
    """Post processing of mask predictions.

    Args:
        mask_threshold (float, optional): Positive threshold. Defaults to 0.5.
        no_overlap (bool, optional): Whether to remove overlapping pixels
            between masks. Defaults to False.
    """

    def __init__(
        self, mask_threshold: float = 0.5, no_overlap: bool = False
    ) -> None:
        """Creates an instance of the class."""
        super().__init__()
        self.mask_threshold = mask_threshold
        self.no_overlap = no_overlap

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
            if self.no_overlap:
                pasted_masks = remove_overlap(pasted_masks, scores)
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


class MaskRCNNHeadLoss(nn.Module):
    """Mask RoI head loss function.

    Args:
        num_classes (int): number of object categories.
    """

    def __init__(self, num_classes: int) -> None:
        """Creates an instance of the class."""
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


def positive_mask_sampler(
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
        [torch.eq(label, 1) for label in sampled_targets.labels],
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
