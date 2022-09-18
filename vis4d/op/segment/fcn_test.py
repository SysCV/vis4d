"""FCN tests."""
import unittest
from typing import Optional, Tuple

import skimage
import torch

from ..base.resnet import ResNet
from ..utils import load_model_checkpoint
from .fcn import FCN, FCNForResNet


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


REV_KEYS = [
    (r"^backbone\.", "body."),
    (r"^aux_classifier\.", "heads.0."),
    (r"^classifier\.", "heads.1."),
]


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
        fcn = FCN(basemodel.out_channels, 21, resize=(512, 512))

        fcn.eval()
        with torch.no_grad():
            features = basemodel(sample_images)
            outputs = fcn(features)

        assert outputs.pred.shape == (2, 21, 512, 512)


class FCNResNetTest(unittest.TestCase):
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
        basemodel = ResNet(
            "resnet50",
            pretrained=True,
            replace_stride_with_dilation=[False, True, True],
        )
        fcn = FCNForResNet(
            basemodel.out_channels,
            21,
            seg_channel_idx=[4, 5],
            resize=(512, 512),
        )

        weights = (
            "https://download.pytorch.org/models/"
            "fcn_resnet50_coco-1167a1af.pth"
        )
        # load_model_checkpoint(basemodel, weights, REV_KEYS)
        # load_model_checkpoint(fcn, weights, REV_KEYS)

        fcn.eval()
        with torch.no_grad():
            features = basemodel(sample_images)
            for feat in features:
                print(feat.shape, end=" ")
            print()
            outputs = fcn(features)

        assert outputs.pred.shape == (2, 21, 512, 512)
