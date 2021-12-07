"""Testcases for mot dataset evaluation."""
import unittest

from scalabel.label.io import load

from vis4d.unittest.utils import get_test_file

from .mot import MOTChallenge, MOTDatasetConfig


class TestMOTEval(unittest.TestCase):
    """build Testcase class."""

    def test_evaluation(self) -> None:
        """Test tracking evaluation in MOT dataset."""
        ref_metrics = {
            "idf1": 0.40281124497991966,
            "idp": 0.5946640316205534,
            "idr": 0.30455465587044533,
            "recall": 0.4896761133603239,
            "precision": 0.9561264822134388,
            "num_unique_objects": 53,
            "mostly_tracked": 10,
            "partially_tracked": 24,
            "mostly_lost": 19,
            "num_false_positives": 222,
            "num_misses": 5042,
            "num_switches": 88,
            "num_fragmentations": 138,
            "mota": 0.4582995951417004,
            "motp": 0.18964631500744994,
            "num_transfer": 27,
            "num_ascend": 65,
            "num_migrate": 8,
        }
        mot_dataset = MOTChallenge(
            MOTDatasetConfig(
                name="mot17_test",
                type="",
                data_root=get_test_file("motchallenge"),
                annotations=get_test_file("motchallenge/result.json"),
            )
        )
        mot_results = load(get_test_file("motchallenge/result.json")).frames
        metrics, _ = mot_dataset.evaluate("track", mot_results, [])
        for k, v in metrics.items():
            self.assertEqual(v, ref_metrics[k])
