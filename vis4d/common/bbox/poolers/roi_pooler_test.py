"""Test cases for torchvision roi pooler."""
import unittest

import torch

from vis4d.struct import Boxes2D
from vis4d.unittest.utils import generate_dets

from .roi_pooler import MultiScaleRoIAlign, MultiScaleRoIPool


class TestMultiScaleRoIPooler(unittest.TestCase):
    """Test cases for multi-scale torchvision roi pooler."""

    def test_pool(self) -> None:
        """Testcase for pool function."""
        pooler = MultiScaleRoIAlign(
            resolution=(7, 7),
            strides=[8, 16],
            sampling_ratio=0,
        )

        N, C, H, W = 2, 128, 1024, 1024  # pylint: disable=invalid-name
        inputs = [
            torch.zeros(
                (N, C, H // pooler.strides[0], W // pooler.strides[0])
            ),
            torch.zeros(
                (N, C, H // pooler.strides[1], W // pooler.strides[1])
            ),
        ]

        boxes_list = []
        for _ in range(N):
            boxes_list += [generate_dets(H, W, 10)]

        out = pooler(inputs, boxes_list)
        self.assertEqual(out.shape, (N * 10, C, 7, 7))

        pooler = MultiScaleRoIPool(
            resolution=(7, 7),
            strides=[8],
            sampling_ratio=0,
        )
        out = pooler([inputs[0]], boxes_list)
        self.assertEqual(out.shape, (N * 10, C, 7, 7))

        boxes_list = []
        for _ in range(N):
            boxes_list += [Boxes2D.empty()]

        out = pooler([inputs[0]], boxes_list)
        self.assertEqual(out.shape, (0, C, 7, 7))

        inputs = [
            torch.zeros((0, C, H // pooler.strides[0], W // pooler.strides[0]))
        ]
        out = pooler(inputs, [])
        self.assertEqual(out.shape, (0, C, 7, 7))
