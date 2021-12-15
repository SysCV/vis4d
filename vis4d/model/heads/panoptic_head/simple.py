"""Simple panoptic head."""
from typing import Tuple

import torch

from vis4d.struct import (
    InputSample,
    InstanceMasks,
    LabelInstances,
    LossesType,
    SemanticMasks,
)

from .base import BasePanopticHead, BasePanopticHeadConfig, PanopticMasks


class SimplePanopticHeadConfig(BasePanopticHeadConfig):
    """Config for simple panoptic head."""

    overlap_thr: float = 0.5
    stuff_area_thr: int = 4096
    thing_conf_thr: float = 0.5


class SimplePanopticHead(BasePanopticHead):
    """Simple panoptic head."""

    def __init__(self, cfg: BasePanopticHeadConfig):
        """Init."""
        super().__init__()
        self.cfg: SimplePanopticHead = SimplePanopticHead(**cfg.dict())

    def forward_train(
        self,
        inputs: InputSample,
        predictions: LabelInstances,
        targets: LabelInstances,
    ) -> LossesType:
        """Forward pass during training stage.

        Returns no losses since simple panoptic head has no learnable
        parameters.
        """
        return {}

    def _combine_segms(
        self, ins_segm: InstanceMasks, sem_segm: SemanticMasks
    ) -> Tuple[InstanceMasks, SemanticMasks]:
        """Combine instance and semantic masks.

        Uses a simple combining logic following
        "combine_semantic_and_instance_predictions.py" in panopticapi.
        """
        # foreground mask
        foreground = torch.zeros(
            ins_segm.masks.shape[1:], dtype=torch.bool, device=self.device
        )

        # sort instance outputs by scores
        sorted_inds = ins_segm.score.argsort(descending=True)

        # add instances one-by-one, check for overlaps with existing ones
        for inst_id in sorted_inds:
            score = ins_segm.score[inst_id].item()
            if score < self.cfg.thing_conf_thr:
                break

            mask = ins_segm.masks[inst_id]  # H,W
            mask_area = mask.sum().item()
            if mask.sum().item() == 0:
                continue

            intersect = torch.logical_and(mask, foreground)
            intersect_area = intersect.sum().item()

            if intersect_area * 1.0 / mask_area > self.cfg.overlap_thr:
                mask[mask > 0] = 0
                continue

            if intersect_area > 0:
                ins_segm.masks[inst_id] = torch.logical_and(mask, ~foreground)
            foreground = torch.logical_and(mask, foreground)

        # add semantic results to remaining empty areas
        for mask in sem_segm.masks:
            # if semantic_label == 0:  # 0 is a special "thing" class
            #     continue
            mask = torch.logical_and(mask, ~foreground)
            if mask.sum().item() < self.cfg.stuff_area_thr:
                mask[mask > 0] = 0

        return ins_segm, sem_segm

    def forward_test(
        self, inputs: InputSample, predictions: LabelInstances
    ) -> PanopticMasks:
        """Forward pass during testing stage."""
        instance_segms, semantic_segms = (
            predictions.instance_masks,
            predictions.semantic_masks,
        )
        assert len(instance_segms) == len(
            semantic_segms
        ), "Length of predictions is not the same, but should be"

        pan_segms = []
        for ins_segm, sem_segm in zip(instance_segms, semantic_segms):
            pan_segms.append(self._combine_segms(ins_segm, sem_segm))

        return [p[0] for p in pan_segms], [p[1] for p in pan_segms]
