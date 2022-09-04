"""FCN tests."""
import unittest
from typing import Optional, Tuple

import skimage
import torch

from ...base.resnet import ResNet
from ...fpp.fcn import FCN
from .fcn_head import FCNHead


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


class FCNTest(unittest.TestCase):
    """FCN test class."""

    def test_inference(self):
        """Test inference of FCN."""
        image1 = url_to_tensor(
            "https://farm1.staticflickr.com/106/311161252_33d75830fd_z.jpg",
            (512, 512),
        )
        image2 = url_to_tensor(
            "https://farm4.staticflickr.com/3217/2980271186_9ec726e0fa_z.jpg",
            (512, 512),
        )
        sample_images = torch.cat([image1, image2])
        basemodel = ResNet("resnet50", pretrained=True, trainable_layers=3)
        fcn = FCN()

        fcn.eval()
        with torch.no_grad():
            features = basemodel(sample_images)
            features = fcn(features)
            out = features[0]

        assert outs.shape == (2, 5, 64, 64)
