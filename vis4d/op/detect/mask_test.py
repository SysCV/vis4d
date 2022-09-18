"""Test cases for Vis4D mask ops."""
import unittest

import torch

from vis4d.unittest.utils import generate_dets

from .mask import paste_masks_in_image


class TestMaskOps(unittest.TestCase):
    """Test cases Vis4D mask ops."""

    def test_paste_masks_in_image(self) -> None:
        """Testcase for paste_masks_in_image."""
        h, w, num_masks, num_dets = 28, 28, 5, 5
        pad_shape = (56, 128)

        masks = torch.rand((num_masks, h, w))
        dets = generate_dets(h, w, num_dets).boxes
        pasted = paste_masks_in_image(masks, dets, pad_shape)
        self.assertEqual((pasted.size(2), pasted.size(1)), pad_shape)
        self.assertEqual(len(pasted), num_masks)
