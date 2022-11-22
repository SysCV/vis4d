"""Test cases for mask utils."""
import unittest

import torch

from vis4d.unittest.util import generate_boxes, generate_masks

from .util import nhw_to_hwc_mask, paste_masks_in_image, postprocess_segms


class TestMaskUtil(unittest.TestCase):
    """Test cases mask utils."""

    def test_paste_masks_in_image(self) -> None:
        """Testcase for paste_masks_in_image."""
        h, w, num_dets = 28, 28, 5
        pad_shape = (56, 128)

        masks = torch.rand((num_dets, h, w))
        dets = generate_boxes(h, w, num_dets, 1)[0][0]
        pasted = paste_masks_in_image(masks, dets, pad_shape)
        self.assertEqual((pasted.size(2), pasted.size(1)), pad_shape)
        self.assertEqual(len(pasted), num_dets)

    def test_nhw_to_hwc_mask(self) -> None:
        """Testcase for nhw_to_hwc_mask."""
        h, w, num_dets, num_cls = 128, 256, 5, 10
        masks = torch.randint(0, 2, (num_dets, h, w))
        classes = torch.randint(0, num_cls, (num_dets,))
        hwc_masks = nhw_to_hwc_mask(masks, classes)
        self.assertEqual(hwc_masks.shape, (h, w))
        for cls_id in classes:
            self.assertTrue(cls_id in hwc_masks.unique())

    def test_postprocess_segms(self) -> None:
        """Testcase for postprocess_segms."""
        batch_size, h, w, num_cls = 2, 128, 256, 5
        h_, w_ = 100, 200
        segms = torch.rand((batch_size, num_cls, h, w))
        post_segms = postprocess_segms(segms, [(h, w)] * 2, [(h_, w_)] * 2)
        self.assertEqual(post_segms.shape, (batch_size, h_, w_))
        self.assertEqual(len(post_segms.unique()), num_cls)
