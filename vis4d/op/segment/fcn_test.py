"""FCN tests."""
from __future__ import annotations

import unittest

import skimage
import torch

from ..base.resnet import ResNet
from .fcn import FCNHead, FCNLoss


def normalize(img: torch.Tensor) -> torch.Tensor:
    """Normalize the image tensor."""
    pixel_mean = (123.675, 116.28, 103.53)
    pixel_std = (58.395, 57.12, 57.375)
    pixel_mean = torch.tensor(pixel_mean, device=img.device).view(-1, 1, 1)
    pixel_std = torch.tensor(pixel_std, device=img.device).view(-1, 1, 1)
    img = (img.float() - pixel_mean) / pixel_std
    return img


def url_to_tensor(
    url: str, im_wh: tuple[int, int] | None = None
) -> torch.Tensor:
    """Load image from URL."""
    image = skimage.io.imread(url)
    if im_wh is not None:
        image = skimage.transform.resize(image, im_wh) * 255
    return normalize(
        torch.tensor(image).float().permute(2, 0, 1).unsqueeze(0).contiguous()
    )


REV_KEYS = [
    (r"^backbone\.", "body."),
    (r"^aux_classifier\.", "heads.0."),
    (r"^classifier\.", "heads.1."),
]


class FCNHeadTest(unittest.TestCase):
    """FCNResNet test class."""

    def test_inference(self):
        """Test inference of FCN with ResNet."""
        image1 = url_to_tensor(
            "https://farm1.staticflickr.com/106/311161252_33d75830fd_z.jpg",
            (512, 512),
        )
        image2 = url_to_tensor(
            "https://farm4.staticflickr.com/3217/2980271186_9ec726e0fa_z.jpg",
            (512, 512),
        )
        sample_images = torch.cat([image1, image2])
        mock_targets = torch.randint(0, 21, (2, 512, 512))

        basemodel = ResNet(
            "resnet50",
            pretrained=True,
            replace_stride_with_dilation=[False, True, True],
        )
        fcn = FCNHead(
            basemodel.out_channels[-2:],
            21,
            resize=(512, 512),
        )
        fcn_loss_weighted = FCNLoss(feature_idx=[4, 5], weights=[0.5, 1])
        fcn_loss_unweighted = FCNLoss(feature_idx=[4, 5], weights=None)

        fcn.eval()
        with torch.no_grad():
            features = basemodel(sample_images)
            pred, outputs = fcn(features)
            losses_weighted = fcn_loss_weighted(outputs, mock_targets)
            losses_unweighted = fcn_loss_unweighted(outputs, mock_targets)

        assert len(outputs) == 6
        assert len(losses_weighted.losses) == 2
        assert len(losses_unweighted.losses) == 2
        assert pred.shape == (2, 21, 512, 512)
        for output in outputs[-2:]:
            assert output.shape == (2, 21, 512, 512)
