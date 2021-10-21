"""Test cases for base augmentation class."""
import unittest

import torch
from scalabel.label.typing import Frame

from vist.struct import Images, InputSample

from . import (
    BaseAugmentation,
    BaseAugmentationConfig,
    KorniaAugmentationWrapper,
)


class TestBaseAugmentation(unittest.TestCase):
    """Test cases VisT base augmentation."""

    def test_generate_parameters(self) -> None:
        """Test generate_parameters function."""
        aug_cfg = BaseAugmentationConfig(type="test", kwargs={})
        base_aug = BaseAugmentation(aug_cfg)
        test_tensor = torch.empty(3, 5)
        params = base_aug.generate_parameters(test_tensor.size())
        self.assertTrue("batch_prob" in params)
        self.assertEqual(params["batch_prob"].size(0), 3)
        base_aug.cfg.prob = 1.0
        params = base_aug.generate_parameters(test_tensor.size())
        self.assertTrue("batch_prob" in params)
        self.assertEqual(params["batch_prob"].size(0), 3)
        self.assertTrue(params["batch_prob"].all())

    def test_call(self) -> None:
        """Test __call__ function."""
        aug_cfg = BaseAugmentationConfig(type="test", kwargs={})
        base_aug = BaseAugmentation(aug_cfg)
        testt = torch.zeros(1, 3, 5, 5)
        params = base_aug.generate_parameters(testt.size())
        test_image = Images(testt, [(testt.shape[3], testt.shape[2])])
        sample = InputSample([Frame(name="test_frame")], test_image)
        results, _ = base_aug(sample, params)
        self.assertTrue(torch.isclose(results.images.tensor, testt).all())
        self.assertEqual(len(results.boxes2d[0].boxes), 0)
        self.assertEqual(len(results.bitmasks[0].masks), 0)


class TestKorniaAugmentation(unittest.TestCase):
    """Test cases Kornia augmentation wrapper."""

    def test_generate_parameters(self) -> None:
        """Test generate_parameters function."""
        aug_cfg = BaseAugmentationConfig(
            type="test", kornia_type="RandomRotation", kwargs={"degrees": 10.0}
        )
        kor_aug = KorniaAugmentationWrapper(aug_cfg)
        num_exps = 2
        testt = torch.zeros(num_exps, 3, 5, 5)
        params = kor_aug.generate_parameters(testt.size())
        self.assertTrue("batch_prob" in params and "degrees" in params)
        self.assertEqual(len(params["batch_prob"]), num_exps)
        self.assertEqual(len(params["degrees"]), num_exps)
        self.assertTrue(params["batch_prob"].all())
        kor_aug.cfg.prob = 0.0
        params = kor_aug.generate_parameters(testt.size())
        self.assertTrue("batch_prob" in params and "degrees" in params)
        self.assertEqual(len(params["batch_prob"]), num_exps)
        self.assertEqual(len(params["degrees"]), 0)
        self.assertTrue(~params["batch_prob"].all())
        kor_aug.cfg.prob = 0.5
        num_exps = 100
        testt = torch.zeros(num_exps, 3, 5, 5)
        params = kor_aug.generate_parameters(testt.size())
        self.assertEqual(params["batch_prob"].sum(), params["degrees"].size(0))

    def test_compute_transformation(self) -> None:
        """Test compute_transformation function."""
        aug_cfg = BaseAugmentationConfig(
            type="test", kornia_type="RandomRotation", kwargs={"degrees": 10.0}
        )
        kor_aug = KorniaAugmentationWrapper(aug_cfg)
        num_exps = 2
        testt = torch.zeros(num_exps, 3, 5, 5)
        test_image = Images(testt, [(testt.shape[3], testt.shape[2])])
        sample = InputSample([Frame(name="test_frame")], test_image)
        params = kor_aug.generate_parameters(testt.size())
        tm = kor_aug.augmentor.compute_transformation(
            sample.images.tensor, params
        )
        self.assertEqual(tm.size(0), num_exps)
        self.assertEqual((tm.size(1), tm.size(2)), (3, 3))

    def test_call(self) -> None:
        """Test __call__ function."""
        aug_cfg = BaseAugmentationConfig(
            type="test", kornia_type="RandomRotation", kwargs={"degrees": 10.0}
        )
        kor_aug = KorniaAugmentationWrapper(aug_cfg)
        kor_aug.cfg.prob = 0.5
        num_exps = 100
        testt = torch.zeros(num_exps, 3, 5, 5)
        test_image = Images(testt, [(testt.shape[3], testt.shape[2])])
        sample = InputSample([Frame(name="test_frame")], test_image)
        params = kor_aug.generate_parameters(testt.size())
        # results, tm = kor_aug(sample, params, False)
        print(sample)
        print(params)  # TODO finish
