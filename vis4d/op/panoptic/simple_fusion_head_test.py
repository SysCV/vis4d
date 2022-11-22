"""Testcases for simple panoptic fusion head."""
import unittest

from vis4d.unittest.util import generate_masks

from ..mask.util import nhw_to_hwc_mask
from .simple_fusion_head import INSTANCE_OFFSET, SimplePanopticFusionHead


class TestSimplePanopticFusionHead(unittest.TestCase):
    """Testcases for SimplePanopticFusionHead."""

    def test_forward(self) -> None:
        """Testcase for forward pass."""
        h, w, num_dets, num_sems = 128, 256, 5, 2
        ins_masks, ins_scores, ins_classes = generate_masks(h, w, num_dets)
        sem_masks, _, sem_classes = generate_masks(h, w, num_sems)
        ins_scores[0][0] = 0.0
        ins_scores[0][1] = 1.0
        ins_scores[0][2:] = 0.9
        ins_masks[0][2] = 0
        ins_masks[0][3][:, :] = ins_masks[0][1][:, :]
        ins_masks[0][4][:16, :16] = ins_masks[0][1][:16, :16]
        sem_masks[0][0][:, :] = 0
        sem_masks[0][1][:, :] = 1
        sem_masks[0] = nhw_to_hwc_mask(sem_masks[0], sem_classes[0])

        pan_head = SimplePanopticFusionHead(
            overlap_thr=0.99, stuff_area_thr=128
        )
        pan_outs = pan_head(ins_masks, ins_scores, ins_classes, sem_masks)
        self.assertEqual(pan_outs.shape, (1, h, w))
        self.assertTrue((pan_outs.unique() >= INSTANCE_OFFSET).sum() > 0)
        pan_ids, pan_cnts = pan_outs.unique(return_counts=True)
        self.assertTrue(len(pan_ids) > 1)
        self.assertTrue(pan_cnts.sum() == h * w)
