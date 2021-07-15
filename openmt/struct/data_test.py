"""Test cases for openMT data structures."""
import unittest

import torch
from scalabel.label.typing import Frame

from .data import Images, InputSample


class TestDataStructures(unittest.TestCase):
    """Test cases openMT data structures."""

    im1 = Images(torch.zeros(2, 1, 128, 128), [(110, 110), (120, 120)])
    im2 = Images(torch.zeros(1, 1, 138, 138), [(130, 130)])

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
        sample = InputSample(Frame(name="f1"), self.im1, image2=self.im2)
        self.assertEqual(sample.metadata.name, "f1")
        meta = sample.get("metadata")
        assert isinstance(meta, Frame)
        self.assertEqual(meta.name, "f1")  # pylint: disable=no-member
        self.assertEqual(sample.image.tensor.shape, self.im1.tensor.shape)
        image2 = sample.get("image2")
        assert isinstance(image2, Images)
        self.assertEqual(image2.tensor.shape, self.im2.tensor.shape)
        self.assertEqual(
            list(sample.dict().keys()), ["metadata", "image", "image2"]
        )
