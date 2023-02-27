"""FCN tests."""
from __future__ import annotations

import unittest

import torch

from tests.util import generate_features
from vis4d.op.base.resnet import ResNet
from vis4d.op.segment.fcn import FCNHead, FCNLoss

REV_KEYS = [
    (r"^backbone\.", "body."),
    (r"^aux_classifier\.", "heads.0."),
    (r"^classifier\.", "heads.1."),
]


class FCNHeadTest(unittest.TestCase):
    """FCNResNet test class."""

    def test_inference(self) -> None:
        """Test inference of FCN with ResNet."""
        test_images = generate_features(3, 512, 512, 1, 2)[0]
        mock_targets = torch.randint(0, 21, (2, 512, 512))

        basemodel = ResNet(
            "resnet50",
            pretrained=True,
            replace_stride_with_dilation=[False, True, True],
        )
        fcn = FCNHead(basemodel.out_channels[-2:], 21, resize=(512, 512))
        fcn_loss_weighted = FCNLoss(feature_idx=[4, 5], weights=[0.5, 1])
        fcn_loss_unweighted = FCNLoss(feature_idx=[4, 5], weights=None)

        fcn.eval()
        with torch.no_grad():
            features = basemodel(test_images)
            pred, outputs = fcn(features)
            losses_weighted = fcn_loss_weighted(outputs, mock_targets)
            losses_unweighted = fcn_loss_unweighted(outputs, mock_targets)

        assert len(outputs) == 6
        assert len(losses_weighted.keys()) == 2
        assert len(losses_unweighted.keys()) == 2
        assert pred.shape == (2, 21, 512, 512)
        for output in outputs[-2:]:
            assert output.shape == (2, 21, 512, 512)
