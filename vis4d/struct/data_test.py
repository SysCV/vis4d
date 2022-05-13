"""Test cases for Vis4D data structures."""
import unittest

import torch

from .data import Extrinsics, Images, Intrinsics, PointCloud

# TODO remove, move relevant parts to new locations (data / common)


class TestDataStructures(unittest.TestCase):
    """Test cases Vis4D data structures."""

    im1 = Images(torch.zeros(2, 1, 128, 128), [(110, 110), (120, 120)])
    im2 = Images(torch.zeros(1, 1, 138, 138), [(130, 130)])

    intr1 = Intrinsics(torch.zeros(3, 3))
    intr2 = Intrinsics(torch.ones(2, 3, 3))

    extr1 = Extrinsics(torch.zeros(4, 4))
    extr2 = Extrinsics(torch.ones(2, 4, 4))

    points1 = PointCloud(torch.zeros(2, 1, 4))
    points2 = PointCloud(torch.zeros(2, 1, 4))

    def test_intrinsics(self) -> None:
        """Testcases for intrinsics class."""
        intrs = Intrinsics.cat([self.intr1, self.intr2])
        self.assertEqual(intrs.tensor.shape, (3, 3, 3))
        ims = intrs.to(torch.device("cpu"))
        self.assertEqual(ims.device, torch.device("cpu"))

    def test_extrinsics(self) -> None:
        """Testcases for extrinsics class."""
        extrs = Extrinsics.cat([self.extr1, self.extr2])
        self.assertEqual(extrs.tensor.shape, (3, 4, 4))
        ims = extrs.to(torch.device("cpu"))
        self.assertEqual(ims.device, torch.device("cpu"))
        self.assertTrue(
            torch.equal(self.extr1.tensor, self.extr1.transpose().tensor)
        )
        self.assertTrue(
            torch.equal((self.extr1 @ self.extr2[0]).tensor, self.extr1.tensor)
        )

    def test_images(self) -> None:
        """Testcases for images class."""
        Images.stride = 16
        ims = Images.cat([self.im1, self.im2])
        self.assertEqual(ims.tensor.shape, (3, 1, 144, 144))
        self.assertEqual(ims[0].tensor.shape, (1, 1, 110, 110))
        ims = ims.to(torch.device("cpu"))
        self.assertEqual(ims.device, torch.device("cpu"))

    def test_points(self) -> None:
        """Testcases for point class."""
        points = PointCloud.cat([self.points1, self.points2])
        self.assertEqual(points.tensor.shape, (4, 1, 4))
        self.assertEqual(points[0].tensor.shape, (1, 1, 4))
        points = points.to(torch.device("cpu"))
        self.assertEqual(points.device, torch.device("cpu"))

    def test_im_resize(self) -> None:
        """Testcase for Images resize."""
        im = Images(torch.zeros(2, 1, 128, 128), [(110, 110), (120, 120)])
        im.resize((256, 256))
        self.assertEqual(im.tensor.shape, (2, 1, 256, 256))
        self.assertEqual(im.image_sizes[0], im.image_sizes[1], (256, 256))

    def test_im_flip(self) -> None:
        """Testcase for Images flip."""
        im = Images(torch.tensor([[[[0, 1], [2, 3]]]]), [(2, 2)])
        im.flip()
        self.assertTrue(
            (im.tensor[0, 0] - torch.tensor([[1, 0], [3, 2]]) == 0).all()
        )
        self.assertEqual(im.image_sizes[0], (2, 2))
