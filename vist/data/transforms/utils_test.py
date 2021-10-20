"""Unit tests for transform utils."""
import unittest

import torch
from torch.distributions import Bernoulli

from .utils import adapted_sampling, batch_prob_generator, identity_matrix


class TestTransformUtils(unittest.TestCase):
    """Test cases for transform utils."""

    def test_identity_matrix(self) -> None:
        """Test identity_matrix function."""
        test_tensor = torch.empty(1)
        im = identity_matrix(test_tensor)
        self.assertEqual(im.size(0), 3)
        self.assertEqual(im.size(1), 3)
        self.assertEqual(test_tensor.device, im.device)
        self.assertEqual(test_tensor.dtype, im.dtype)

    def test_adapted_sampling(self) -> None:
        """Test adapted_sampling function."""
        batch_prob = adapted_sampling((5,), Bernoulli(1.0), False).bool()
        self.assertEqual(len(batch_prob), 5)
        batch_prob = adapted_sampling((1,), Bernoulli(1.0), True).bool()
        self.assertEqual(len(batch_prob), 1)

    def test_batch_prob_generator(self) -> None:
        """Test batch_prob_generator function."""
        test_tensor = torch.empty(3, 4)
        batch_prob = batch_prob_generator(test_tensor.size(), 1.0, 1.0, False)
        self.assertEqual(len(batch_prob), 3)
        self.assertTrue(batch_prob.all())
        batch_prob = batch_prob_generator(test_tensor.size(), 0.0, 1.0, False)
        self.assertEqual(len(batch_prob), 3)
        self.assertTrue((~batch_prob).all())
        batch_prob = batch_prob_generator(test_tensor.size(), 1.0, 0.0, False)
        self.assertEqual(len(batch_prob), 3)
        self.assertTrue((~batch_prob).all())
        batch_prob = batch_prob_generator(test_tensor.size(), 0.0, 0.0, False)
        self.assertEqual(len(batch_prob), 3)
        self.assertTrue((~batch_prob).all())
        batch_prob = batch_prob_generator(test_tensor.size(), 0.5, 0.5, False)
        self.assertEqual(len(batch_prob), 3)
        batch_prob = batch_prob_generator(test_tensor.size(), 0.5, 1.0, False)
        self.assertEqual(len(batch_prob), 3)
