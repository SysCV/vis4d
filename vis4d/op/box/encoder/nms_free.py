"""NMS-Free bounding box coder."""
from __future__ import annotations

import torch
from torch import Tensor


# TODO: Add Encoder
class NMSFreeDecoder:
    """BBox decoder for NMS-free detector."""

    def __init__(
        self,
        num_classes: int = 10,
        post_center_range: list[float] | None = None,
        max_num: int = 100,
        score_threshold: float | None = None,
    ) -> None:
        """Initialize NMSFreeDecoder.

        Args:
            post_center_range (list[float]): Limit of the center.
                Default: None.
            max_num (int): Max number to be kept. Default: 100.
            score_threshold (float): Threshold to filter boxes based on score.
                Default: None.
            code_size (int): Code size of bboxes. Default: 9
        """
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes

    def _decode_single(
        self, cls_scores: Tensor, bbox_preds: Tensor
    ) -> dict[str, Tensor]:
        """Decode single batch bboxes.

        Args:
            cls_scores (Tensor): Outputs from the classification head, in shape
                of [num_query, cls_out_channels]. Note cls_out_channels
                should includes background.
            bbox_preds (Tensor): Outputs from the regression
                head with normalized coordinate format (cx, cy, w, l, cz, h,
                rot_sine, rot_cosine, vx, vy). Shape [num_query, 9].

        Returns:
            dict[str, Tensor]: Decoded boxes.
        """
        max_num = self.max_num

        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.view(-1).topk(max_num)
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

        if self.post_center_range is not None:
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
            predictions_dict = {
                "bboxes": boxes3d,
                "scores": scores,
                "labels": labels,
            }

        else:
            raise NotImplementedError(
                "Need to reorganize output as a batch, only "
                "support post_center_range is not None for now!"
            )
        return predictions_dict

    def decode(
        self, preds_dicts: dict[str, Tensor]
    ) -> list[dict[str, Tensor]]:
        """Decode bboxes.

        Args:
            all_cls_scores (Tensor): Outputs from the classification head,
                shape [nb_dec, bs, num_query, cls_out_channels]. Note
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression
                head with normalized coordinate format (cx, cy, w, l, cz, h,
                rot_sine, rot_cosine, vx, vy). Shape [nb_dec, bs, num_query,
                9].

        Returns:
            list[dict]: Decoded boxes.
        """
        all_cls_scores = preds_dicts["all_cls_scores"][-1]
        all_bbox_preds = preds_dicts["all_bbox_preds"][-1]

        batch_size = all_cls_scores.size()[0]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(
                self._decode_single(all_cls_scores[i], all_bbox_preds[i])
            )
        return predictions_list


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
