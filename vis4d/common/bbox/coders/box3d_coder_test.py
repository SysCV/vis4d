"""Test cases for Box3D coder."""
import unittest

import torch

from vis4d.struct import Boxes2D, Intrinsics

from .box3d_coder import QD3DTBox3DCoder, QD3DTBox3DCoderConfig


class TestBox3DCoder(unittest.TestCase):
    """Test cases for Box3D coder."""

    coder = QD3DTBox3DCoder(QD3DTBox3DCoderConfig(type="abc"))

    def test_decode_empty(self) -> None:
        """Test decode function when input is empty."""
        result = self.coder.decode(
            [Boxes2D(torch.empty(0, 5), torch.empty(0), torch.empty(0))],
            [torch.empty(0)],
            Intrinsics(torch.eye(3)),
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 0)
