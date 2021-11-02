"""Test cases for augmentations."""
import copy
import unittest

from vis4d.unittest.utils import generate_input_sample

from .augmentations import RandomCrop, RandomCropConfig, Resize, ResizeConfig


class TestResize(unittest.TestCase):
    """Test cases Vis4D Resize."""

    def test_generate_parameters(self) -> None:
        """Test generate_parameters function."""
        aug_cfg = ResizeConfig(type="test", shape=(5, 5))
        resize = Resize(aug_cfg)
        num_imgs, num_objs, height, width = 1, 10, 10, 10
        sample = generate_input_sample(height, width, num_imgs, num_objs)
        params = resize.generate_parameters(sample)
        self.assertTrue("apply" in params and "shape" in params)
        self.assertEqual(params["apply"].size(0), 1)
        self.assertTrue(params["apply"].item())
        self.assertEqual(tuple(params["transform"].shape), (1, 3, 3))
        resize.cfg.prob = 0.0
        params = resize.generate_parameters(sample)
        self.assertTrue("apply" in params and "shape" in params)
        self.assertEqual(params["apply"].size(0), 1)
        self.assertFalse(params["apply"].item())
        sample = generate_input_sample(height, width, 10, num_objs)
        resize.cfg.prob = 1.0
        params = resize.generate_parameters(sample)
        self.assertTrue("apply" in params and "shape" in params)
        self.assertEqual(params["apply"].size(0), 10)
        self.assertTrue(params["apply"].all())
        self.assertEqual(tuple(params["transform"].shape), (10, 3, 3))

    def test_call(self) -> None:
        """Test __call__ function."""
        aug_cfg = ResizeConfig(type="test", shape=(5, 5))
        resize = Resize(aug_cfg)
        num_imgs, num_objs, height, width = 1, 10, 10, 10
        sample = generate_input_sample(height, width, num_imgs, num_objs)
        pre_intrs = copy.deepcopy(sample.intrinsics.tensor)
        pre_boxes = copy.deepcopy(sample.boxes2d[0].boxes)
        pre_masks = copy.deepcopy(sample.instance_masks[0].masks)
        results, _ = resize(sample, None)
        self.assertEqual(sample, results)
        self.assertEqual(tuple(results.boxes2d[0].boxes.shape), (num_objs, 5))
        self.assertEqual(
            tuple(results.instance_masks[0].masks.shape), (num_objs, 5, 5)
        )
        new_intrs = sample.intrinsics.tensor
        new_boxes = sample.boxes2d[0].boxes
        new_masks = sample.instance_masks[0].masks
        self.assertEqual(pre_boxes.shape, new_boxes.shape)
        self.assertNotEqual(pre_masks.shape, new_masks.shape)
        self.assertFalse((pre_intrs == new_intrs).all())
        resize.cfg.prob = 0.0
        _, _ = resize(sample, None)
        sample = generate_input_sample(height, width, 2, num_objs)
        _, _ = resize(sample, None)
        self.assertEqual(len(sample.boxes2d), 2)
        self.assertEqual(len(sample.instance_masks), 2)


class TestRandomCrop(unittest.TestCase):
    """Test cases Vis4D RandomCrop."""

    def test_generate_parameters(self) -> None:
        """Test generate_parameters function."""
        aug_cfg = RandomCropConfig(type="test", shape=(5, 5))
        crop = RandomCrop(aug_cfg)
        num_imgs, num_objs, height, width = 1, 10, 10, 10
        sample = generate_input_sample(height, width, num_imgs, num_objs)
        params = crop.generate_parameters(sample)
        self.assertTrue(
            tuple(params.keys())
            == ("apply", "image_wh", "crop_params", "keep")
        )
        self.assertEqual(params["apply"].size(0), 1)
        self.assertTrue(params["apply"].item())
        crop.cfg.prob = 0.0
        params = crop.generate_parameters(sample)
        self.assertTrue(
            tuple(params.keys())
            == ("apply", "image_wh", "crop_params", "keep")
        )
        self.assertEqual(params["apply"].size(0), 1)
        self.assertFalse(params["apply"].item())
        sample = generate_input_sample(height, width, 10, num_objs)
        crop.cfg.prob = 1.0
        params = crop.generate_parameters(sample)
        self.assertTrue(
            tuple(params.keys())
            == ("apply", "image_wh", "crop_params", "keep")
        )
        self.assertEqual(params["apply"].size(0), 10)
        self.assertTrue(params["apply"].all())
        self.assertEqual(tuple(params["crop_params"].shape), (10, 4))
        self.assertEqual(len(params["keep"]), 10)

    def test_call(self) -> None:
        """Test __call__ function."""
        aug_cfg = RandomCropConfig(
            type="test",
            shape=[(1, 1), (2, 2)],
            crop_type="absolute_range",
            allow_empty_crops=False,
        )
        crop = RandomCrop(aug_cfg)
        num_imgs, num_objs, height, width = 1, 10, 10, 10
        sample = generate_input_sample(height, width, num_imgs, num_objs)
        pre_intrs = copy.deepcopy(sample.intrinsics.tensor)
        results, params = crop(sample, None)
        self.assertEqual(len(sample), len(results))
        new_intrs = sample.intrinsics.tensor
        new_boxes = sample.boxes2d[0].boxes
        new_masks = sample.instance_masks[0].masks
        self.assertEqual(new_boxes.shape[0], params["keep"][0].sum().item())
        self.assertEqual(new_masks.shape[0], params["keep"][0].sum().item())
        self.assertFalse((pre_intrs == new_intrs).all())
        crop.cfg.prob = 0.0
        _, _ = crop(sample, None)
        sample = generate_input_sample(height, width, 2, num_objs)
        _, _ = crop(sample, None)
        self.assertEqual(len(sample.boxes2d), 2)
        self.assertEqual(len(sample.instance_masks), 2)
