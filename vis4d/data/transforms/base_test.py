"""Test cases for base augmentation class."""
import copy
import unittest

import torch

from vis4d.unittest.utils import generate_input_sample

from . import BaseAugmentation, BaseAugmentationConfig


class TestBaseAugmentation(unittest.TestCase):
    """Test cases Vis4D base augmentation."""

    def test_generate_parameters(self) -> None:
        """Test generate_parameters function."""
        aug_cfg = BaseAugmentationConfig(type="test", kwargs={})
        base_aug = BaseAugmentation(aug_cfg)
        num_imgs, num_objs, height, width = 3, 10, 5, 5
        sample = generate_input_sample(height, width, num_imgs, num_objs)
        params = base_aug.generate_parameters(sample)
        self.assertTrue("apply" in params)
        self.assertEqual(params["apply"].size(0), 3)
        self.assertTrue(params["apply"].all())
        base_aug.cfg.prob = 0.0
        params = base_aug.generate_parameters(sample)
        self.assertTrue("apply" in params)
        self.assertEqual(params["apply"].size(0), 3)

    def test_call(self) -> None:
        """Test __call__ function."""
        aug_cfg = BaseAugmentationConfig(type="test", kwargs={})
        base_aug = BaseAugmentation(aug_cfg)
        num_imgs, num_objs, height, width = 3, 10, 5, 5
        sample = generate_input_sample(height, width, num_imgs, num_objs)
        pre_image = copy.deepcopy(sample.images.tensor)
        results, _ = base_aug(sample, None)
        self.assertTrue(torch.isclose(results.images.tensor, pre_image).all())
        self.assertEqual(len(results.targets.boxes2d[0].boxes), num_objs)
        self.assertEqual(
            len(results.targets.instance_masks[0].masks), num_objs
        )
