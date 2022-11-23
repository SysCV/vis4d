"""Track visualziation test cases."""
import unittest

import numpy as np
import torch

from vis4d.unittest.utils import generate_dets, generate_dets3d

from ..struct_to_revise import Intrinsics
from .track import draw_sequence
from .util import preprocess_image


class TestTrackVis(unittest.TestCase):
    """Tests for vis4d.vis.track."""

    def test_draw_sequence(self) -> None:
        """Test draw sequence function."""
        seq_imgs = [torch.rand(3, 128, 128) for _ in range(10)]
        proc_imgs = [preprocess_image(im) for im in seq_imgs]
        seq_dets = [
            generate_dets(128, 128, 10, track_ids=True) for _ in range(10)
        ]
        seq_dets3d = [generate_dets3d(10, track_ids=True) for _ in range(10)]
        intrinsics = [Intrinsics(torch.eye(3)) for _ in range(10)]
        seq = draw_sequence(seq_imgs, seq_dets, seq_dets3d, intrinsics)

        self.assertEqual(len(seq), 10)

        for orig_frame, frame in zip(proc_imgs, seq):
            self.assertEqual(np.array(frame).shape, (128, 128, 3))
            # check if frame was modified
            self.assertFalse((np.array(orig_frame) == np.array(frame)).all())
