"""SimOTA label assigner.

Modified from mmdetection (https://github.com/open-mmlab/mmdetection).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from vis4d.op.box.box2d import bbox_iou

from .base import MatchResult

INF = 100000.0
EPS = 1.0e-7


class SimOTAMatcher(nn.Module):
    """SimOTA label assigner used by YOLOX.

    Args:
        center_radius (float, optional): Ground truth center size to judge
            whether a prior is in center. Defaults to 2.5.
        candidate_topk (int, optional): The candidate top-k which used to
            get top-k ious to calculate dynamic-k. Defaults to 10.
        iou_weight (float, optional): The scale factor for regression
            iou cost. Defaults to 3.0.
        cls_weight (float, optional): The scale factor for classification
            cost. Defaults to 1.0.
    """

    def __init__(
        self,
        center_radius: float = 2.5,
        candidate_topk: int = 10,
        iou_weight: float = 3.0,
        cls_weight: float = 1.0,
    ):
        """Init."""
        super().__init__()
        self.center_radius = center_radius
        self.candidate_topk = candidate_topk
        self.iou_weight = iou_weight
        self.cls_weight = cls_weight

    def forward(  # pylint: disable=arguments-differ # type: ignore[override]
        self,
        pred_scores: Tensor,
        priors: Tensor,
        decoded_bboxes: Tensor,
        gt_bboxes: Tensor,
        gt_labels: Tensor,
    ) -> MatchResult:
        """Assign gt to priors using SimOTA.

        Args:
            pred_scores (Tensor): Classification scores of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            decoded_bboxes (Tensor): Predicted bboxes, a 2D-Tensor with shape
                [num_priors, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].

        Returns:
            MatchResult: The assigned result.
        """
        num_gt = gt_bboxes.size(0)
        num_bboxes = decoded_bboxes.size(0)

        # assign 0 by default
        assigned_gt_inds = decoded_bboxes.new_full(
            (num_bboxes,), 0, dtype=torch.long
        )
        valid_mask, is_in_boxes_and_center = self.get_in_gt_and_in_center_info(
            priors, gt_bboxes
        )
        valid_decoded_bbox = decoded_bboxes[valid_mask]
        valid_pred_scores = pred_scores[valid_mask]
        num_valid = valid_decoded_bbox.size(0)

        if num_gt == 0 or num_bboxes == 0 or num_valid == 0:
            # No ground truth or boxes, return empty assignment
            assigned_gt_iou = decoded_bboxes.new_zeros((num_bboxes,))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = decoded_bboxes.new_full(
                    (num_bboxes,), -1, dtype=torch.long
                )
            return MatchResult(
                assigned_gt_indices=assigned_gt_inds,
                assigned_labels=assigned_labels,
                assigned_gt_iou=assigned_gt_iou,
            )

        pairwise_ious = bbox_iou(valid_decoded_bbox, gt_bboxes)
        iou_cost = -torch.log(pairwise_ious + EPS)

        gt_onehot_label = (
            F.one_hot(  # pylint: disable=not-callable
                gt_labels.to(torch.int64), pred_scores.shape[-1]
            )
            .float()
            .unsqueeze(0)
            .repeat(num_valid, 1, 1)
        )

        valid_pred_scores = valid_pred_scores.unsqueeze(1).repeat(1, num_gt, 1)
        # disable AMP autocast and calculate BCE with FP32 to avoid overflow
        with torch.cuda.amp.autocast(enabled=False):
            cls_cost = (
                F.binary_cross_entropy(
                    valid_pred_scores.to(dtype=torch.float32),
                    gt_onehot_label,
                    reduction="none",
                )
                .sum(-1)
                .to(dtype=valid_pred_scores.dtype)
            )

        cost_matrix = (
            cls_cost * self.cls_weight
            + iou_cost * self.iou_weight
            + (~is_in_boxes_and_center) * INF
        )

        matched_pred_ious, matched_gt_inds = self.dynamic_k_matching(
            cost_matrix, pairwise_ious, num_gt, valid_mask
        )

        # convert to MatchResult format
        assigned_gt_inds[valid_mask] = matched_gt_inds
        assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
        assigned_labels[valid_mask] = 1
        assigned_gt_iou = assigned_gt_inds.new_full(
            (num_bboxes,), -INF, dtype=torch.float32
        )
        assigned_gt_iou[valid_mask] = matched_pred_ious
        return MatchResult(
            assigned_gt_indices=assigned_gt_inds,
            assigned_labels=assigned_labels,
            assigned_gt_iou=assigned_gt_iou,
        )

    def get_in_gt_and_in_center_info(
        self, priors: Tensor, gt_bboxes: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Get whether the priors are in gt bboxes and in centers."""
        num_gt = gt_bboxes.size(0)

        repeated_x = priors[:, 0].unsqueeze(1).repeat(1, num_gt)
        repeated_y = priors[:, 1].unsqueeze(1).repeat(1, num_gt)
        repeated_stride_x = priors[:, 2].unsqueeze(1).repeat(1, num_gt)
        repeated_stride_y = priors[:, 3].unsqueeze(1).repeat(1, num_gt)

        # is prior centers in gt bboxes, shape: [n_prior, n_gt]
        l_ = repeated_x - gt_bboxes[:, 0]
        t_ = repeated_y - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - repeated_x
        b_ = gt_bboxes[:, 3] - repeated_y

        deltas = torch.stack([l_, t_, r_, b_], dim=1)
        is_in_gts = deltas.min(dim=1).values > 0
        is_in_gts_all = is_in_gts.sum(dim=1) > 0

        # is prior centers in gt centers
        gt_cxs = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        gt_cys = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
        ct_box_l = gt_cxs - self.center_radius * repeated_stride_x
        ct_box_t = gt_cys - self.center_radius * repeated_stride_y
        ct_box_r = gt_cxs + self.center_radius * repeated_stride_x
        ct_box_b = gt_cys + self.center_radius * repeated_stride_y

        cl_ = repeated_x - ct_box_l
        ct_ = repeated_y - ct_box_t
        cr_ = ct_box_r - repeated_x
        cb_ = ct_box_b - repeated_y

        ct_deltas = torch.stack([cl_, ct_, cr_, cb_], dim=1)
        is_in_cts = ct_deltas.min(dim=1).values > 0
        is_in_cts_all = is_in_cts.sum(dim=1) > 0

        # in boxes or in centers, shape: [num_priors]
        is_in_gts_or_centers = is_in_gts_all | is_in_cts_all

        # both in boxes and centers, shape: [num_fg, num_gt]
        is_in_boxes_and_centers = (
            is_in_gts[is_in_gts_or_centers, :]
            & is_in_cts[is_in_gts_or_centers, :]
        )
        return is_in_gts_or_centers, is_in_boxes_and_centers

    def dynamic_k_matching(
        self,
        cost: Tensor,
        pairwise_ious: Tensor,
        num_gt: int,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Dynamic K matching strategy."""
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
        # select candidate topk ious for dynamic-k calculation
        candidate_topk = min(self.candidate_topk, pairwise_ious.size(0))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)
        # calculate dynamic k for each gt
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[:, gt_idx],
                k=dynamic_ks[gt_idx].item(),  # type: ignore
                largest=False,
            )
            matching_matrix[:, gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        prior_match_gt_mask = matching_matrix.sum(1) > 1
        if prior_match_gt_mask.sum() > 0:
            _, cost_argmin = torch.min(cost[prior_match_gt_mask, :], dim=1)
            matching_matrix[prior_match_gt_mask, :] *= 0
            matching_matrix[prior_match_gt_mask, cost_argmin] = 1
        # get foreground mask inside box and center prior
        fg_mask_inboxes = matching_matrix.sum(1) > 0
        valid_mask[valid_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(1)
        matched_pred_ious = (matching_matrix * pairwise_ious).sum(1)[
            fg_mask_inboxes
        ]
        return matched_pred_ious, matched_gt_inds

    def __call__(
        self,
        pred_scores: Tensor,
        priors: Tensor,
        decoded_bboxes: Tensor,
        gt_bboxes: Tensor,
        gt_labels: Tensor,
    ) -> MatchResult:
        """Type declaration for forward."""
        return self._call_impl(
            pred_scores, priors, decoded_bboxes, gt_bboxes, gt_labels
        )
