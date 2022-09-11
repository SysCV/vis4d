"""Test cases for augmentations."""
import copy
import unittest

import torch

from vis4d.struct import InputSample
from vis4d.unittest.utils import generate_input_sample

from .augmentations import MixUp, Mosaic, RandomCrop, Resize


class TestResize(unittest.TestCase):
    """Test cases Vis4D Resize."""

    def test_generate_parameters(self) -> None:
        """Test generate_parameters function."""
        resize = Resize(shape=(5, 5))
        num_imgs, num_objs, height, width = 1, 10, 10, 10
        sample = generate_input_sample(height, width, num_imgs, num_objs)
        params = resize.generate_parameters(sample)
        self.assertTrue("apply" in params and "shape" in params)
        self.assertEqual(params["apply"].size(0), 1)
        self.assertTrue(params["apply"].item())
        self.assertEqual(tuple(params["transform"].shape), (1, 3, 3))
        resize.prob = 0.0
        params = resize.generate_parameters(sample)
        self.assertTrue("apply" in params and "shape" in params)
        self.assertEqual(params["apply"].size(0), 1)
        self.assertFalse(params["apply"].item())
        sample = generate_input_sample(height, width, 10, num_objs)
        resize.prob = 1.0
        params = resize.generate_parameters(sample)
        self.assertTrue("apply" in params and "shape" in params)
        self.assertEqual(params["apply"].size(0), 10)
        self.assertTrue(params["apply"].all())
        self.assertEqual(tuple(params["transform"].shape), (10, 3, 3))

    def test_call(self) -> None:
        """Test __call__ function."""
        resize = Resize(shape=(5, 5))
        num_imgs, num_objs, height, width = 1, 10, 10, 9
        sample = generate_input_sample(height, width, num_imgs, num_objs)
        pre_intrs = copy.deepcopy(sample.intrinsics.tensor)
        pre_tgts = copy.deepcopy(sample.targets)
        pre_boxes = pre_tgts.boxes2d[0].boxes
        pre_masks = pre_tgts.instance_masks[0].masks
        results, _ = resize(sample, None)
        self.assertEqual(sample, results)
        new_tgts = results.targets
        self.assertEqual(tuple(new_tgts.boxes2d[0].boxes.shape), (num_objs, 5))
        self.assertEqual(
            tuple(new_tgts.instance_masks[0].masks.shape), (num_objs, 5, 5)
        )
        new_intrs = sample.intrinsics.tensor
        new_boxes = new_tgts.boxes2d[0].boxes
        new_masks = new_tgts.instance_masks[0].masks
        self.assertEqual(pre_boxes.shape, new_boxes.shape)
        self.assertNotEqual(pre_masks.shape, new_masks.shape)
        self.assertFalse((pre_intrs == new_intrs).all())
        resize.prob = 0.0
        _, _ = resize(sample, None)
        sample = generate_input_sample(height, width, 2, num_objs)
        _, _ = resize(sample, None)
        self.assertEqual(len(sample.targets.boxes2d), 2)
        self.assertEqual(len(sample.targets.instance_masks), 2)

    def test_multiscale_list(self) -> None:
        """Test multiscale list mode."""
        shapes = [(5, 5), (10, 10)]
        resize = Resize(shape=shapes, multiscale_mode="list")
        num_imgs, num_objs, height, width = 1, 20, 20, 9
        sample = generate_input_sample(height, width, num_imgs, num_objs)
        sample, _ = resize(sample, None)
        self.assertTrue(sample.images.image_sizes[0] in shapes)


class TestRandomCrop(unittest.TestCase):
    """Test cases Vis4D RandomCrop."""

    def test_generate_parameters(self) -> None:
        """Test generate_parameters function."""
        crop = RandomCrop(shape=(5, 5))
        num_imgs, num_objs, height, width = 1, 10, 10, 10
        sample = generate_input_sample(height, width, num_imgs, num_objs)
        params = crop.generate_parameters(sample)
        self.assertTrue(
            tuple(params.keys())
            == ("apply", "image_wh", "crop_params", "keep")
        )
        self.assertEqual(params["apply"].size(0), 1)
        self.assertTrue(params["apply"].item())
        crop.prob = 0.0
        params = crop.generate_parameters(sample)
        self.assertTrue(
            tuple(params.keys())
            == ("apply", "image_wh", "crop_params", "keep")
        )
        self.assertEqual(params["apply"].size(0), 1)
        self.assertFalse(params["apply"].item())
        sample = generate_input_sample(height, width, 10, num_objs)
        crop.prob = 1.0
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
        crop = RandomCrop(
            shape=[(1, 1), (2, 2)],
            crop_type="absolute_range",
            allow_empty_crops=False,
            recompute_boxes2d=True,
        )
        num_imgs, num_objs, height, width = 1, 10, 10, 10
        sample = generate_input_sample(height, width, num_imgs, num_objs)
        pre_intrs = copy.deepcopy(sample.intrinsics.tensor)
        results, params = crop(sample, None)
        self.assertEqual(len(sample), len(results))
        new_intrs = sample.intrinsics.tensor
        new_boxes = sample.targets.boxes2d[0].boxes
        new_masks = sample.targets.instance_masks[0].masks
        self.assertEqual(new_boxes.shape[0], params["keep"][0].sum().item())
        self.assertEqual(new_masks.shape[0], params["keep"][0].sum().item())
        self.assertFalse((pre_intrs == new_intrs).all())
        crop.prob = 0.0
        _, _ = crop(sample, None)
        sample = generate_input_sample(height, width, 2, num_objs)
        _, _ = crop(sample, None)
        self.assertEqual(len(sample.targets.boxes2d), 2)
        self.assertEqual(len(sample.targets.instance_masks), 2)
        # segmentation masks
        crop.prob = 1.0
        sample = generate_input_sample(
            height, width, num_imgs, num_objs, det_input=False
        )
        results, params = crop(sample, None)
        new_masks = sample.targets.semantic_masks[0].masks
        self.assertEqual(new_masks.shape[0], num_objs)
        crop = RandomCrop(shape=(2, 2), cat_max_ratio=0.2)
        num_objs = 2
        sample = generate_input_sample(
            height, width, num_imgs, num_objs, det_input=False
        )
        results, params = crop(sample, None)
        new_masks = sample.targets.semantic_masks[0].masks
        self.assertEqual(new_masks.shape[0], num_objs)
        self.assertEqual(new_masks.shape[1:], (2, 2))

    def test_relative(self) -> None:
        """Test relative crop option."""
        crop = RandomCrop(shape=(0.5, 0.5), crop_type="relative")
        num_imgs, num_objs, height, width = 1, 10, 10, 10
        sample = generate_input_sample(height, width, num_imgs, num_objs)
        sample, _ = crop(sample, None)
        self.assertEqual(sample.images.image_sizes[0], (5, 5))

    def test_relative_range(self) -> None:
        """Test relative range crop option."""
        crop = RandomCrop(
            shape=[(0.1, 0.1), (0.5, 0.5)], crop_type="relative_range"
        )
        num_imgs, num_objs, height, width = 1, 10, 10, 10
        sample = generate_input_sample(height, width, num_imgs, num_objs)
        sample, _ = crop(sample, None)
        self.assertTrue(1 <= sample.images.image_sizes[0][0] <= 5)
        self.assertTrue(1 <= sample.images.image_sizes[0][1] <= 5)


class TestMosaic(unittest.TestCase):
    """Test cases Vis4D Mosaic."""

    def test_mosaic(self) -> None:
        """Test mosaic augmentation."""
        mosaic = Mosaic(out_shape=(10, 10), pad_value=0.0)
        num_imgs, num_objs, height, width = 1, 10, 10, 9
        samples = []
        for _ in range(4):
            samples += [
                generate_input_sample(
                    height, width, num_imgs, num_objs, track_ids=True
                )
            ]
        sample = InputSample.cat(samples)
        sample.images.tensor = torch.zeros_like(sample.images.tensor)
        out, _ = mosaic(sample)
        self.assertEqual(len(out.images.image_sizes), 1)
        self.assertEqual(out.images.image_sizes[0], (20, 20))
        self.assertTrue((out.images.tensor == 0.0).all())
        self.assertTrue(
            len(out.targets.boxes2d[0])
            <= sum([len(s.targets.boxes2d[0]) for s in samples])
        )


class TestMixUp(unittest.TestCase):
    """Test cases Vis4D MixUp."""

    def test_mixup(self) -> None:
        """Test mixup augmentation."""
        mixup = MixUp(out_shape=(10, 10), ratio_range=(1.2, 1.5))
        num_imgs, num_objs, height, width = 1, 10, 10, 9
        samples = [
            generate_input_sample(
                height, width, num_imgs, num_objs, track_ids=True
            )
        ]
        samples += [
            generate_input_sample(
                height, width, num_imgs, num_objs, track_ids=True
            )
        ]
        sample = InputSample.cat(samples)
        out, _ = mixup(sample)
        self.assertEqual(len(out.images.image_sizes), 1)
        self.assertEqual(out.images.image_sizes[0], (20, 20))
        self.assertTrue(
            len(out.targets.boxes2d[0])
            <= len(samples[0].targets.boxes2d[0])
            + len(samples[1].targets.boxes2d[0])
        )
