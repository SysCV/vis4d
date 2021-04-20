"""Test cases for tracking engine prediction."""
from openmt.track.predict import predict, track_predict_func
from openmt.unittest.util import TrackTest, d2_data_reset


class TestPredict(TrackTest):
    """Test cases for openmt tracking prediction."""

    def test_predict(self) -> None:
        """Testcase for predict function."""
        self.assertIsNotNone(self.det2cfg)
        self.assertIsNotNone(self.cfg)
        results = track_predict_func(self.det2cfg, self.cfg)
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

    def test_predict_launcher(self) -> None:
        """Testcase for prediction launcher."""
        d2_data_reset(self.det2cfg.DATASETS.TRAIN)
        d2_data_reset(self.det2cfg.DATASETS.TEST)
        predict(self.cfg)
