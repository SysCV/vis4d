"""Test cases for detection engine prediction."""
from openmt.detect.predict import predict
from openmt.unittest.util import DetectTest


class TestPredict(DetectTest):
    """Test cases for openmt detection prediction."""

    def test_predict(self) -> None:
        """Testcase for predict function."""
        self.assertIsNotNone(self.cfg)
        results = predict(self.cfg)

        metric_keys = [
            "AP",
            "AP50",
            "AP75",
            "APs",
            "APm",
            "APl",
            "AP-pedestrian",
            "AP-rider",
            "AP-car",
            "AP-truck",
            "AP-bus",
            "AP-train",
            "AP-motorcycle",
            "AP-bicycle",
            "AP-traffic light",
            "AP-traffic sign",
        ]

        for k in results["bbox"]:
            self.assertIn(k, metric_keys)
