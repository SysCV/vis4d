"""Test cases for Box3D coder."""
import unittest

import torch

from vis4d.struct_to_revise import Boxes2D, Intrinsics

from .box3d import QD3DTBox3DCoder


# TODO fix test after fixing box3d.py, renaming for now
class TestBox3DCoder(unittest.TestCase):
    """Test cases for Box3D coder."""

    coder = QD3DTBox3DCoder()

    def test_decode_empty(self) -> None:
        """Test decode function when input is empty."""
        result = self.coder.decode(
            [Boxes2D.empty()], [torch.empty(0)], Intrinsics(torch.eye(3))
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 0)
