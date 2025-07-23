"""Track assignment functions."""

from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor


def greedy_assign(
    detection_scores: Tensor,
    tracklet_ids: Tensor,
    affinity_scores: Tensor,
    match_score_thr: float = 0.5,
    obj_score_thr: float = 0.3,
    nms_conf_thr: None | float = None,
) -> Tensor:
    """Greedy assignment of detections to tracks given affinities."""
    ids = torch.full(
        (len(detection_scores),),
        -1,
        dtype=torch.long,
        device=detection_scores.device,
    )

    for i, score in enumerate(detection_scores):
        conf, memo_ind = torch.max(affinity_scores[i, :], dim=0)
        cur_id = tracklet_ids[memo_ind]
        if conf > match_score_thr:
            if cur_id > -1:
                if score > obj_score_thr:
                    ids[i] = cur_id
                    affinity_scores[:i, memo_ind] = 0
                    affinity_scores[(i + 1) :, memo_ind] = 0
                elif nms_conf_thr is not None and conf > nms_conf_thr:
                    ids[i] = -2
    return ids


def hungarian_assign(
    detection_scores: Tensor,
    tracklet_ids: Tensor,
    affinity_scores: Tensor,
    match_score_thr: float = 0.5,
    obj_score_thr: float = 0.3,
    nms_conf_thr: None | float = None,
) -> Tensor:
    """Hungarian assignment of detections to tracks given affinities."""
    ids = torch.full(
        (len(detection_scores),),
        -1,
        dtype=torch.long,
        device=detection_scores.device,
    )

    matched_indices = linear_sum_assignment(-affinity_scores.cpu().numpy())

    for idx in range(len(matched_indices[0])):
        i = matched_indices[0][idx]
        memo_ind = matched_indices[1][idx]
        conf = affinity_scores[i, memo_ind]
        tid = tracklet_ids[memo_ind]
        if conf > match_score_thr and tid > -1:
            if detection_scores[i] > obj_score_thr:
                ids[i] = tid
                affinity_scores[:i, memo_ind] = 0
                affinity_scores[i + 1 :, memo_ind] = 0
            elif nms_conf_thr is not None and conf > nms_conf_thr:
                ids[i] = -2

    return ids


class TrackIDCounter:
    """Global counter for track ids.

    Holds a count of tracks to enable unique and contiguous track ids starting
    from zero.
    """

    count: int = 0

    @classmethod
    def reset(cls) -> None:
        """Reset track id counter."""
        cls.count = 0

    @classmethod
    def get_ids(
        cls, num_ids: int, device: torch.device = torch.device("cpu")
    ) -> Tensor:
        """Generate a num_ids number of new unique tracking ids.

        Args:
            num_ids (int): number of ids
            device (torch.device, optional): Device to create ids on. Defaults
                to torch.device("cpu").

        Returns:
            Tensor: Tensor of new contiguous track ids.
        """
        new_ids = torch.arange(cls.count, cls.count + num_ids, device=device)
        cls.count = cls.count + num_ids
        return new_ids
