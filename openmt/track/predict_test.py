"""Test cases for tracking engine prediction."""
from openmt.track.predict import predict
from openmt.unittest.util import TrackTest


class TestPredict(TrackTest):
    """Test cases for openmt tracking prediction."""

    def test_predict(self) -> None:
        """Testcase for predict function."""
        self.assertIsNotNone(self.cfg)
        self.cfg.launch.action = "predict"
        results = predict(self.cfg)
        metric_keys = [
            "pedestrian",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
            "HUMAN",
            "VEHICLE",
            "BIKE",
            "OVERALL",
        ]
        for k in results:
            self.assertIn(k, metric_keys)
