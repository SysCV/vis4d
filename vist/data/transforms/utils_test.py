"""Unit tests for transform utils."""
import unittest

from .utils import sample_batched, sample_bernoulli


class TestTransformUtils(unittest.TestCase):
    """Test cases for transform utils."""

    def test_sample_bernoulli(self) -> None:
        """Test sample_bernoulli function."""
        batch_prob = sample_bernoulli(5, 1.0)
        self.assertEqual(len(batch_prob), 5)
        batch_prob = sample_bernoulli(1, 1.0)
        self.assertEqual(len(batch_prob), 1)

    def test_sample_batched(self) -> None:
        """Test sample_batched function."""
        batch_prob = sample_batched(3, 1.0, False)
        self.assertEqual(len(batch_prob), 3)
        self.assertTrue(batch_prob.all())
        batch_prob = sample_batched(3, 0.0, False)
        self.assertEqual(len(batch_prob), 3)
        self.assertTrue((~batch_prob).all())
        batch_prob = sample_batched(3, 1.0, False)
        self.assertEqual(len(batch_prob), 3)
        self.assertTrue(batch_prob.all())
        batch_prob = sample_batched(3, 0.0, False)
        self.assertEqual(len(batch_prob), 3)
        self.assertTrue((~batch_prob).all())
        batch_prob = sample_batched(3, 0.5, False)
        self.assertEqual(len(batch_prob), 3)
        batch_prob = sample_batched(3, 0.5, False)
        self.assertEqual(len(batch_prob), 3)
        batch_prob = sample_batched(3, 1.0, True)
        self.assertEqual(len(batch_prob), 3)
        self.assertTrue((batch_prob).all())
        batch_prob = sample_batched(3, 0.0, True)
        self.assertEqual(len(batch_prob), 3)
        self.assertTrue((~batch_prob).all())
