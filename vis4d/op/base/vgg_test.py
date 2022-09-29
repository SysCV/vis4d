"""Testcases for DLA backbone."""
import unittest
from typing import Optional, Tuple

import skimage
import torch

from .vgg import VGG


def normalize(img: torch.Tensor) -> torch.Tensor:
    pixel_mean = (123.675, 116.28, 103.53)
    pixel_std = (58.395, 57.12, 57.375)
    pixel_mean = torch.tensor(pixel_mean, device=img.device).view(-1, 1, 1)
    pixel_std = torch.tensor(pixel_std, device=img.device).view(-1, 1, 1)
    img = (img.float() - pixel_mean) / pixel_std
    return img


def url_to_tensor(
    url: str, im_wh: Optional[Tuple[int, int]] = None
) -> torch.Tensor:
    image = skimage.io.imread(url)
    if im_wh is not None:
        image = skimage.transform.resize(image, im_wh) * 255
    return normalize(
        torch.tensor(image).float().permute(2, 0, 1).unsqueeze(0).contiguous()
    )


class TestVGG(unittest.TestCase):
    """Testcases for VGG backbone."""

    def test_vgg(self) -> None:
        image1 = url_to_tensor(
            "https://farm1.staticflickr.com/106/311161252_33d75830fd_z.jpg",
            (512, 512),
        )
        image2 = url_to_tensor(
            "https://farm4.staticflickr.com/3217/2980271186_9ec726e0fa_z.jpg",
            (512, 512),
        )
        sample_images = torch.cat([image1, image2])

        for vgg_name in ["vgg11", "vgg13", "vgg16", "vgg19"]:
            self._test_vgg(vgg_name, sample_images)
            self._test_vgg(vgg_name + "_bn", sample_images)

    def _test_vgg(self, vgg_name: str, sample_images: torch.Tensor) -> None:
        """Testcase for VGG."""

        vgg = VGG(vgg_name, pretrained=False)
        out = vgg(sample_images)

        channels = [3, 3, 64, 128, 256, 512, 512]
        self.assertEqual(vgg.out_channels, channels)
        self.assertEqual(len(out), 7)

        self.assertEqual(out[0].shape[0], 2)
        self.assertEqual(out[0].shape[1], 3)
        self.assertEqual(out[0].shape[2], 512)
        self.assertEqual(out[0].shape[3], 512)

        for i in range(1, 7):
            feat = out[i]
            self.assertEqual(feat.shape[0], 2)
            self.assertEqual(feat.shape[1], channels[i])
            self.assertEqual(feat.shape[2], 512 / (2 ** (i - 1)))
            self.assertEqual(feat.shape[3], 512 / (2 ** (i - 1)))
