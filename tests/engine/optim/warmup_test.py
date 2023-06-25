"""Test cases for warmup."""
import unittest

from vis4d.engine.optim import (
    ConstantLRWarmup,
    ExponentialLRWarmup,
    QuadraticLRWarmup,
)


class TestWarmup(unittest.TestCase):
    """Test cases Vis4D warmup."""

    def test_constant(self) -> None:
        """Test case for constant LR warmup."""
        warmup = ConstantLRWarmup(0.5, 5)
        lr = warmup(2, 0.1, 0.0)
        self.assertTrue(isinstance(lr, float))
        self.assertEqual(lr, 0.1 * 0.5)

    def test_exp(self) -> None:
        """Test case for exponential LR warmup."""
        warmup = ExponentialLRWarmup(0.5, 5)
        lr = warmup(2, 0.1, 0.0)
        self.assertTrue(isinstance(lr, float))
        self.assertAlmostEqual(lr, 0.5 ** (1 - 2 / 5) * 0.1)

    def test_quad(self) -> None:
        """Test case for quadratic LR warmup."""
        warmup = QuadraticLRWarmup(0.5, 5)
        lr = warmup(2, 0.1, 0.0)
        self.assertTrue(isinstance(lr, float))
        self.assertAlmostEqual(lr, 0.5 * (2 * 2 + 1) / 5**2 * 0.1)
