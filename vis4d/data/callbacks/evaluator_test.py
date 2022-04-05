"""Evaluator callback tests."""
import unittest

from .evaluator import DefaultEvaluatorCallback


class TestDefaultEvaluatorCallback(unittest.TestCase):
    """Test cases for DefaultEvaluatorCallback."""

    def test_evaluate(self) -> None:
        """Test evaluation."""
        dataset_loader = ...
        evaluator = DefaultEvaluatorCallback(0, dataset_loader)

        # TODO continue
