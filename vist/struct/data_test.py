"""Test cases for VisT data structures."""
import unittest

import torch
from scalabel.label.typing import Frame

from .data import Extrinsics, Images, InputSample, Intrinsics


class TestDataStructures(unittest.TestCase):
    """Test cases VisT data structures."""

    im1 = Images(torch.zeros(2, 1, 128, 128), [(110, 110), (120, 120)])
    im2 = Images(torch.zeros(1, 1, 138, 138), [(130, 130)])

    intr1 = Intrinsics(torch.zeros(3, 3))
    intr2 = Intrinsics(torch.ones(2, 3, 3))

    extr1 = Extrinsics(torch.zeros(4, 4))
    extr2 = Extrinsics(torch.ones(2, 4, 4))

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

    def test_images(self) -> None:
        """Testcases for images class."""
        Images.stride = 16
        ims = Images.cat([self.im1, self.im2])
        self.assertEqual(ims.tensor.shape, (3, 1, 144, 144))
        self.assertEqual(ims[0].tensor.shape, (1, 1, 110, 110))
        ims = ims.to(torch.device("cpu"))
        self.assertEqual(ims.device, torch.device("cpu"))

    def test_inputsample(self) -> None:
        """Testcases for InputSample class."""
        attributes = [
            "metadata",
            "images",
            "boxes2d",
            "boxes3d",
            "intrinsics",
            "extrinsics",
        ]
        sample = InputSample([Frame(name="f1")], self.im2)
        meta = sample.get("metadata")[0]
        assert isinstance(meta, Frame)
        self.assertEqual(meta.name, "f1")
        self.assertEqual(sample.images.tensor.shape, self.im2.tensor.shape)
        self.assertEqual(list(sample.dict().keys()), attributes)
        for attr in attributes:
            sample.get(attr)
