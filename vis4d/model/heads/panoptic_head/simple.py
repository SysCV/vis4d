"""Simple panoptic head."""
from typing import List, Tuple, Union

import torch

from vis4d.struct import (
    InputSample,
    InstanceMasks,
    LabelInstances,
    LossesType,
    SemanticMasks,
)

from .base import BasePanopticHead, PanopticMasks
from .utils import prune_instance_masks


class SimplePanopticHead(BasePanopticHead):
    """Simple panoptic head."""

    def __init__(
        self,
        ignore_class: Union[int, List[int]] = -1,
        overlap_thr: float = 0.5,
        stuff_area_thr: int = 4096,
        thing_conf_thr: float = 0.5,
    ):
        """Init."""
        super().__init__()
        self.overlap_thr = overlap_thr
        self.stuff_area_thr = stuff_area_thr
        self.thing_conf_thr = thing_conf_thr
        if isinstance(ignore_class, int):
            ignore_class = [ignore_class]
        self.ignore_class = ignore_class

    def forward_train(
        self,
        inputs: InputSample,
        predictions: LabelInstances,
        targets: LabelInstances,
    ) -> LossesType:  # pragma: no cover
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
        foreground, _ = prune_instance_masks(
            ins_segm, self.thing_conf_thr, self.overlap_thr
        )

        # add semantic results to remaining empty areas
        for i, (mask, cls_id) in enumerate(
            zip(sem_segm.masks, sem_segm.class_ids)
        ):
            if (
                cls_id in self.ignore_class
                or mask.sum().item() < self.stuff_area_thr
            ):
                mask[mask > 0] = 0
                continue
            sem_segm.masks[i] = torch.logical_and(mask, ~foreground)

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
