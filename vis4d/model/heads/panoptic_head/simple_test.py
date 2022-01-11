"""Testcases for SimplePanopticHead."""
import unittest

from vis4d.unittest.utils import generate_input_sample, generate_semantic_masks

from .simple import SimplePanopticHead, SimplePanopticHeadConfig


class TestSimplePanopticHead(unittest.TestCase):
    """Testcases for SimplePanopticHead."""

    def test_forward(self) -> None:
        """Testcase for forward pass."""
        inputs = generate_input_sample(32, 32, 1, 5)
        inputs.targets.semantic_masks = [generate_semantic_masks(32, 32, 2)]
        ins_masks, sem_masks = (
            inputs.targets.instance_masks[0],
            inputs.targets.semantic_masks[0],
        )
        ins_masks.score[0] = 0.0
        ins_masks.score[1] = 1.0
        ins_masks.score[2:] = 0.9
        ins_masks.masks[2] = 0
        ins_masks.masks[3][:, :] = ins_masks.masks[1][:, :]
        ins_masks.masks[4][:16, :16] = ins_masks.masks[1][:16, :16]
        sem_masks.masks[0][:, :] = 0
        sem_masks.masks[1][:, :] = 1

        cfg = SimplePanopticHeadConfig(
            type="SimplePanopticHead", overlap_thr=0.99, stuff_area_thr=128
        )
        pan_head = SimplePanopticHead(cfg)
        ins_outs, sem_outs = pan_head(inputs, inputs.targets)
        self.assertEqual(len(ins_outs[0]), 5)
        self.assertEqual(len(sem_outs[0]), 2)
        self.assertEqual(ins_outs[0].masks[0].sum(), 0)
        self.assertEqual(ins_outs[0].masks[2].sum(), 0)
        self.assertEqual(ins_outs[0].masks[3].sum(), 0)
        self.assertEqual(sem_outs[0].masks[0].sum(), 0)
        self.assertEqual(ins_outs[0].masks[1].sum(), ins_masks.masks[1].sum())
