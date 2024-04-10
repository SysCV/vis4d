"""NMS-Free bounding box coder for BEVFormer."""

from __future__ import annotations

import torch
from torch import Tensor


class NMSFreeDecoder:
    """BBox decoder for NMS-free detector."""

    def __init__(
        self,
        num_classes: int,
        post_center_range: list[float],
        max_num: int = 100,
        score_threshold: float | None = None,
    ) -> None:
        """Initialize NMSFreeDecoder.

        Args:
            num_classes (int): Number of classes.
            post_center_range (list[float]): Limit of the center.
            max_num (int): Max number to be kept. Default: 100.
            score_threshold (float): Threshold to filter boxes based on score.
                Default: None.
        """
        self.num_classes = num_classes
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold

    def __call__(
        self, cls_scores: Tensor, bbox_preds: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Decode single batch bboxes.

        Args:
            cls_scores (Tensor): Outputs from the classification head, in shape
                of [num_query, cls_out_channels]. Note cls_out_channels
                should includes background.
            bbox_preds (Tensor): Outputs from the regression
                head with normalized coordinate format (cx, cy, w, l, cz, h,
                rot_sine, rot_cosine, vx, vy). Shape [num_query, 9].

        Returns:
            tuple[Tensor, Tensor, Tensor]: Decoded boxes (x, y, z, l, w, h,
                yaw, vx, vy), scores and labels.
        """
        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.view(-1).topk(self.max_num)
        labels = indexs % self.num_classes
        bbox_index = indexs // self.num_classes
        bbox_preds = bbox_preds[bbox_index]

        final_box_preds = _denormalize_bbox(bbox_preds)
        final_scores = scores
        final_preds = labels

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
            tmp_score = self.score_threshold
            while thresh_mask.sum() == 0:
                tmp_score *= 0.9
                if tmp_score < 0.01:
                    thresh_mask = final_scores > -1
                    break
                thresh_mask = final_scores >= tmp_score

        post_center_range = torch.tensor(
            self.post_center_range, device=scores.device
        )
        mask = (final_box_preds[..., :3] >= post_center_range[:3]).all(1)
        mask &= (final_box_preds[..., :3] <= post_center_range[3:]).all(1)

        if self.score_threshold:
            mask &= thresh_mask

        boxes3d = final_box_preds[mask]
        scores = final_scores[mask]

        labels = final_preds[mask]

        return boxes3d, scores, labels


def _denormalize_bbox(normalized_bboxes: Tensor) -> Tensor:
    """Denormalize bounding boxes."""
    # rotation
    rot_sine = normalized_bboxes[..., 6:7]

    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    # center in the bev
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]

    # size
    w = normalized_bboxes[..., 2:3]
    l = normalized_bboxes[..., 3:4]
    h = normalized_bboxes[..., 5:6]

    w = w.exp()
    l = l.exp()
    h = h.exp()
    if normalized_bboxes.size(-1) > 8:
        # velocity
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]
        denormalized_bboxes = torch.cat(
            [cx, cy, cz, w, l, h, rot, vx, vy], dim=-1
        )
    else:
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)

    return denormalized_bboxes
