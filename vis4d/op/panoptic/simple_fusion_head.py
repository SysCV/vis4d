"""Simple panoptic head."""
from __future__ import annotations

import torch
from torch import nn

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
    ) -> None:
        """Creates an instance of the class.

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
            ignore_class = num_things_classes + self.num_stuff_classes
        if isinstance(ignore_class, int):
            self.ignore_class = [ignore_class]
        else:
            self.ignore_class = ignore_class

    def _combine_segms(
        self,
        ins_mask: torch.Tensor,
        ins_score: torch.Tensor,
        ins_class_id: torch.Tensor,
        sem_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Combine instance and semantic masks.

        Uses a simple combining logic following
        "combine_semantic_and_instance_predictions.py" in panopticapi.

        Args:
            ins_mask (torch.Tensor): Instance mask with shape [N, H, W].
            ins_score (torch.Tensor): Instance scores with shape [N].
            ins_class_id (torch.Tensor): Instance class IDs with shape [N].
            sem_mask (torch.Tensor): Semantic mask with shape [H, W].

        Returns:
            torch.Tensor: Panoptic mask with shape [H, W].
        """
        # panoptic segmentation
        pan_segm = torch.zeros(
            ins_mask.shape[1:], dtype=torch.int, device=ins_mask.device
        )

        # sort instance outputs by scores
        sorted_inds = ins_score.argsort(descending=True)

        # add instances one-by-one, check for overlaps with existing ones
        ins_id = 1
        for inst_id in sorted_inds:
            mask = ins_mask[inst_id]  # H,W
            score = ins_score[inst_id].item()
            cls_id = ins_class_id[inst_id].item()
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
        sem_mask[pan_segm > 0] = self.ignore_class[0]
        sem_clses, sem_cnts = sem_mask.unique(return_counts=True)
        for sem_cls, sem_cnt in zip(sem_clses, sem_cnts):
            if sem_cls in self.ignore_class or sem_cnt < self.stuff_area_thr:
                continue
            pan_segm[sem_mask == sem_cls] = sem_cls + self.num_things_classes

        return pan_segm

    def forward(
        self,
        ins_masks: list[torch.Tensor],
        ins_scores: list[torch.Tensor],
        ins_class_ids: list[torch.Tensor],
        sem_masks: list[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            ins_masks (list[torch.Tensor]): List of instance masks, each with
                shape [N, H, W].
            ins_scores (list[torch.Tensor]): List of instance scores, each with
                shape [N].
            ins_class_ids (list[torch.Tensor]): List of class IDs, each with
                shape [N].
            sem_masks (list[torch.Tensor]): List of semantic masks, each with
                shape [H, W].

        Returns:
            torch.Tensor: Panoptic masks with shape [B, H, W].
        """
        ins_masks_list = [
            (ins_masks[i], ins_scores[i], ins_class_ids[i])
            for i in range(len(ins_masks))
        ]
        assert len(ins_masks_list) == len(
            sem_masks
        ), "Length of predictions is not the same, but should be"
        pan_segms = []
        for ins_segm, sem_segm in zip(ins_masks_list, sem_masks):
            pan_segms.append(self._combine_segms(*ins_segm, sem_segm))
        return torch.stack(pan_segms)

    def __call__(
        self,
        ins_masks: list[torch.Tensor],
        ins_scores: list[torch.Tensor],
        ins_class_ids: list[torch.Tensor],
        sem_masks: list[torch.Tensor],
    ) -> torch.Tensor:
        """Type definition for function call."""
        return self._call_impl(ins_masks, ins_scores, ins_class_ids, sem_masks)
