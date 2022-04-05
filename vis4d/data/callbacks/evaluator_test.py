"""Evaluator callback tests."""
import unittest

from vis4d.unittest.utils import generate_input_sample

from ..datasets import Scalabel
from .evaluator import DefaultEvaluatorCallback


class TestDefaultEvaluatorCallback(unittest.TestCase):
    """Test cases for DefaultEvaluatorCallback."""

    def test_evaluate(self) -> None:
        """Test evaluation."""
        base_dir = "vis4d/engine/testcases/track/bdd100k-samples"
        dataset_loader = Scalabel(
            "test_dataset",
            f"{base_dir}/images",
            f"{base_dir}/labels",
            config_path=f"{base_dir}/config.toml",
            eval_metrics=["detect"],
        )
        evaluator = DefaultEvaluatorCallback(0, dataset_loader, "unittests")

        def my_log(key, value, rank_zero_only: bool) -> None:
            print(key, value, rank_zero_only)

        evaluator.log = my_log

        frames = dataset_loader.frames
        for frame in frames:
            assert frame.labels is not None
            for label in frame.labels:
                label.score = 1.0
        test_inputs = [
            [generate_input_sample(28, 28, 1, 4, frame_name=f"test_frame{i}")]
            for i in range(len(frames))
        ]
        evaluator.process(test_inputs, {"detect": [f.labels for f in frames]})
        results = evaluator.evaluate(0)
        self.assertTrue(isinstance(results, dict))
        self.assertGreater(len(results), 0)
