"""Test cases for deepsort tracking graph construction."""
import unittest

import torch

from .res_layer import ResidualBlock


class TestResLayer(unittest.TestCase):
    """Test cases for residual block."""

    def test_dim(self) -> None:
        """Testcase for input dim and output dim the same."""
        c_in, c_out = 256, 256
        block = ResidualBlock(c_in, c_out)
        bs, c_in, height, width = 2, 256, 16, 8
        input_tensor = torch.rand([bs, c_in, height, width])
        out = block(input_tensor)
        self.assertEqual(out.shape, (bs, c_out, height, width))

    def test_dim_downsampling(self) -> None:
        """Testcase for input dim and output dim not the same."""
        c_in, c_out = 256, 128
        block = ResidualBlock(c_in, c_out)
        bs, c_in, height, width = 2, 256, 16, 8
        input_tensor = torch.rand([bs, c_in, height, width])
        out = block(input_tensor)
        self.assertEqual(out.shape, (bs, c_out, height, width))
