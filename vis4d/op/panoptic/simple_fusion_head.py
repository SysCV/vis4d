"""Simple panoptic head."""
from __future__ import annotations

import torch
from torch import nn

from vis4d.op.detect.rcnn import MaskOut

INSTANCE_OFFSET = 1000


class SimplePanopticFusionHead(nn.Module):
    """Simple panoptic fusion head."""

    def __init__(
        self,
        num_things_classes: int = 80,
        num_stuff_classes: int = 53,
        ignore_class: int | list[int] = -1,
        overlap_thr: float = 0.5,
        stuff_area_thr: int = 4096,
        thing_conf_thr: float = 0.5,
    ):
        """Init.

        Args:
            num_things_classes (int, optional): Number of thing (foreground)
                classes. Defaults to 80.
            num_stuff_classes (int, optional): Number of stuff (background)
                classes. Defaults to 53.
            ignore_class (int | list[int], optional): Ignored stuff class.
                Defaults to -1.
            overlap_thr (float, optional): Maximum overlap of thing classes.
                Defaults to 0.5.
            stuff_area_thr (int, optional): Maximum overlap of stuff classes.
                Defaults to 4096.
            thing_conf_thr (float, optional): Minimum confidence threshold.
                Defaults to 0.5.
        """
        super().__init__()
        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        self.overlap_thr = overlap_thr
        self.stuff_area_thr = stuff_area_thr
        self.thing_conf_thr = thing_conf_thr
        if ignore_class == -1:
            ignore_class = self.num_stuff_classes
        if isinstance(ignore_class, int):
            ignore_class = [ignore_class]
        self.ignore_class = ignore_class

    def _combine_segms(
        self, ins_segm: MaskOut, sem_segm: torch.Tensor
    ) -> torch.Tensor:
        """Combine instance and semantic masks.

        Uses a simple combining logic following
        "combine_semantic_and_instance_predictions.py" in panopticapi.
        """
        # panoptic segmentation
        pan_segm = torch.zeros(
            ins_segm.masks[0].shape[1:],
            dtype=torch.int,
            device=ins_segm.masks[0].device,
        )

        # sort instance outputs by scores
        sorted_inds = ins_segm.scores[0].argsort(descending=True)

        # add instances one-by-one, check for overlaps with existing ones
        ins_id = 1
        for inst_id in sorted_inds:
            mask = ins_segm.masks[0][inst_id]  # H,W
            score = ins_segm.scores[0][inst_id].item()
            cls_id = ins_segm.class_ids[0][inst_id].item()
            if score < self.thing_conf_thr:
                continue

            mask_area = mask.sum().item()
            if mask_area == 0:
                continue

            intersect = torch.logical_and(mask, pan_segm != 0)
            intersect_area = intersect.sum().item()
            if intersect_area * 1.0 / mask_area > self.overlap_thr:
                continue
            if intersect_area > 0:
                mask = torch.logical_and(mask, pan_segm == 0)

            pan_segm[mask != 0] = cls_id + ins_id * INSTANCE_OFFSET
            ins_id += 1

        # add semantic results to remaining empty areas
        sem_segm[pan_segm > 0] = self.ignore_class[0]
        sem_clses, sem_cnts = sem_segm.unique(return_counts=True)
        for sem_cls, sem_cnt in zip(sem_clses, sem_cnts):
            if sem_cls in self.ignore_class or sem_cnt < self.stuff_area_thr:
                continue
            pan_segm[sem_segm == sem_cls] = sem_cls + self.num_things_classes

        return pan_segm

    def forward(
        self, ins_masks: MaskOut, sem_masks: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass."""
        ins_masks_list = [
            MaskOut(
                masks=[ins_masks.masks[i]],
                scores=[ins_masks.scores[i]],
                class_ids=[ins_masks.class_ids[i]],
            )
            for i in range(len(ins_masks.masks))
        ]
        assert len(ins_masks_list) == len(
            sem_masks
        ), "Length of predictions is not the same, but should be"
        pan_segms = []
        for ins_segm, sem_segm in zip(ins_masks_list, sem_masks):
            pan_segms.append(self._combine_segms(ins_segm, sem_segm))
        return torch.stack(pan_segms)

    def __call__(
        self, ins_masks: MaskOut, sem_masks: torch.Tensor
    ) -> torch.Tensor:
        """Type definition for function call."""
        return self._call_impl(ins_masks, sem_masks)
