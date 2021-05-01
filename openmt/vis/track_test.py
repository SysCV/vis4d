"""Track visualziation test cases."""
import unittest

import numpy as np
import torch

from openmt.unittest.utils import generate_dets

from .track import draw_sequence
from .utils import preprocess_image


class TestTrackVis(unittest.TestCase):
    """Tests for openmt.vis.track."""

    def test_draw_sequence(self) -> None:
        """Test draw sequence function."""
        seq_imgs = [torch.rand(3, 128, 128) for _ in range(10)]
        proc_imgs = [preprocess_image(im) for im in seq_imgs]
        seq_dets = [
            generate_dets(128, 128, 10, track_ids=True) for _ in range(10)
        ]
        seq = draw_sequence(seq_imgs, seq_dets)

        self.assertEqual(len(seq), 10)

        for orig_frame, frame in zip(proc_imgs, seq):
            self.assertEqual(np.array(frame).shape, (128, 128, 3))
            # check if frame was modified
            self.assertFalse((np.array(orig_frame) == np.array(frame)).all())
