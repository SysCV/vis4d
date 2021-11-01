"""Test cases for Kornia augmentation class."""
import copy
import unittest

from vist.unittest.utils import generate_input_sample

from . import KorniaAugmentationConfig, KorniaAugmentationWrapper


class TestKorniaAugmentation(unittest.TestCase):
    """Test cases Kornia augmentation wrapper."""

    def test_generate_parameters(self) -> None:
        """Test generate_parameters function."""
        aug_cfg = KorniaAugmentationConfig(
            type="test", kornia_type="RandomRotation", kwargs={"degrees": 10.0}
        )
        kor_aug = KorniaAugmentationWrapper(aug_cfg)
        num_imgs, num_objs, height, width = 2, 10, 5, 5
        sample = generate_input_sample(height, width, num_imgs, num_objs)
        params = kor_aug.generate_parameters(sample)
        self.assertTrue("apply" in params and "degrees" in params)
        self.assertTrue("batch_prob" in params and "transform" in params)
        self.assertEqual(len(params["batch_prob"]), num_imgs)
        self.assertEqual(len(params["degrees"]), num_imgs)
        self.assertEqual(len(params["apply"]), num_imgs)
        self.assertTrue(params["batch_prob"].all())
        self.assertTrue(params["apply"].all())
        kor_aug.cfg.prob = 0.0
        params = kor_aug.generate_parameters(sample)
        self.assertTrue("batch_prob" in params and "degrees" in params)
        self.assertEqual(len(params["batch_prob"]), num_imgs)
        self.assertEqual(len(params["degrees"]), 2)
        self.assertFalse(params["batch_prob"].any())
        self.assertFalse(params["apply"].any())
        kor_aug.cfg.prob = 0.5
        num_imgs = 100
        sample = generate_input_sample(height, width, num_imgs, num_objs)
        params = kor_aug.generate_parameters(sample)
        self.assertGreater(params["batch_prob"].sum(), 0)

    def test_compute_transformation(self) -> None:
        """Test compute_transformation function."""
        aug_cfg = KorniaAugmentationConfig(
            type="test", kornia_type="RandomRotation", kwargs={"degrees": 10.0}
        )
        kor_aug = KorniaAugmentationWrapper(aug_cfg)
        num_imgs, num_objs, height, width = 2, 10, 5, 5
        sample = generate_input_sample(height, width, num_imgs, num_objs)
        params = kor_aug.generate_parameters(sample)
        tm = params["transform"]
        self.assertEqual(tm.size(0), num_imgs)
        self.assertEqual((tm.size(1), tm.size(2)), (3, 3))

    def test_call(self) -> None:
        """Test __call__ function."""
        aug_cfg = KorniaAugmentationConfig(
            type="test", kornia_type="RandomRotation", kwargs={"degrees": 10.0}
        )
        kor_aug = KorniaAugmentationWrapper(aug_cfg)
        kor_aug.cfg.prob = 1.0
        num_imgs, num_objs, height, width = 1, 10, 5, 5
        sample = generate_input_sample(height, width, num_imgs, num_objs)
        pre_intrs = copy.deepcopy(sample.intrinsics.tensor)
        pre_boxes = copy.deepcopy(sample.boxes2d[0].boxes)
        pre_masks = copy.deepcopy(sample.instance_masks[0].masks)
        results, _ = kor_aug(sample, None)
        self.assertEqual(sample, results)
        self.assertEqual(tuple(results.boxes2d[0].boxes.shape), (num_objs, 5))
        self.assertEqual(
            tuple(results.instance_masks[0].masks.shape),
            (num_objs, width, height),
        )
        new_intrs = sample.intrinsics.tensor
        new_boxes = sample.boxes2d[0].boxes
        new_masks = sample.instance_masks[0].masks
        self.assertEqual(pre_boxes.shape, new_boxes.shape)
        self.assertEqual(pre_masks.shape, new_masks.shape)
        self.assertFalse((pre_intrs == new_intrs).all())
        aug_cfg = KorniaAugmentationConfig(
            type="test", kornia_type="RandomHorizontalFlip", kwargs={}
        )
        kor_aug = KorniaAugmentationWrapper(aug_cfg)
        kor_aug.cfg.prob = 1.0
        num_imgs = 3
        sample = generate_input_sample(height, width, num_imgs, num_objs)
        results, _ = kor_aug(sample, None)
        self.assertEqual(len(sample.images.tensor), num_imgs)
        self.assertEqual(len(sample.boxes2d), num_imgs)
        self.assertEqual(len(sample.instance_masks), num_imgs)
