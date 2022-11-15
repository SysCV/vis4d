"""Test of the U-Net Architecture."""
import unittest
import torch

from vis4d.op.segment.unet import UNet


class UNetTest(unittest.TestCase):
    """UNetTest test class."""

    bs = 12
    img_res = 128
    n_classes = 10
    depth = 5

    def test_shapes(self):
        """Checks that the network can be called and has the right shapes."""
        net = UNet(num_classes=self.n_classes, in_channels=3, depth=self.depth)
        in_data = torch.rand((self.bs, 3, self.img_res, self.img_res))
        out = net(in_data)
        # Check logits
        self.assertEqual(out.logits.shape[0], self.bs)
        self.assertEqual(out.logits.shape[1], self.n_classes)
        self.assertEqual(out.logits.shape[2], self.img_res)
        self.assertEqual(out.logits.shape[3], self.img_res)

        # Check intermediate features
        self.assertEqual(len(out.intermediate_features), net.depth - 1)

        # Check num channels of intermediate features
        for idx, feature in enumerate(out.intermediate_features):
            self.assertEqual(
                feature.shape[1],
                net.start_filts * (2 ** (net.depth - idx - 2)),
            )
