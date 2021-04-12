"""Modeling utils."""
import math
import random
from typing import List, Tuple

import torch
from detectron2.structures import Instances

from openmt.struct import Boxes2D


def select_keyframe(
    sequence_length: int, strategy: str = "random"
) -> Tuple[int, List[int]]:
    """Keyframe selection.

    Strategies:
    - Random
    - First frame
    - Last frame
    """
    if strategy == "random":
        key_index = random.randint(0, sequence_length - 1)
    elif strategy == "first":
        key_index = 0
    elif strategy == "last":
        key_index = sequence_length - 1
    else:
        raise NotImplementedError(
            f"Keyframe selection strategy {strategy} not implemented"
        )

    ref_indices = list(range(sequence_length))
    ref_indices.remove(key_index)

    return key_index, ref_indices


def detections_to_box2d(detections: List[Instances]) -> List[Boxes2D]:
    """Convert d2 Instances representing detections to Boxes2D."""
    result = []
    for detection in detections:
        boxes, scores, cls = (
            detection.pred_boxes.tensor,
            detection.scores,
            detection.pred_classes,
        )
        result.append(
            Boxes2D(
                torch.cat([boxes, scores.unsqueeze(-1)], -1),
                class_ids=cls,
                image_wh=detection.image_size,
            )
        )
    return result


def proposal_to_box2d(proposals: List[Instances]) -> List[Boxes2D]:
    """Convert d2 Instances representing proposals to Boxes2D."""
    result = []
    for proposal in proposals:
        boxes, logits = (
            proposal.proposal_boxes.tensor,
            proposal.objectness_logits,
        )
        result.append(
            Boxes2D(
                torch.cat([boxes, logits.unsqueeze(-1)], -1),
                image_wh=proposal.image_size,
            )
        )
    return result


def target_to_box2d(
    targets: List[Instances], score_as_logit: bool = True
) -> List[Boxes2D]:
    """Convert d2 Instances representing targets to Boxes2D."""
    result = []
    for target in targets:
        boxes, cls, track_ids = (
            target.gt_boxes.tensor,
            target.gt_classes,
            target.track_ids,
        )
        score = torch.ones((boxes.shape[0], 1), device=boxes.device)
        if score_as_logit:
            score *= math.log((1.0 - 1e-10) / (1 - (1.0 - 1e-10)))
        result.append(Boxes2D(torch.cat([boxes, score], -1), cls, track_ids))
    return result
