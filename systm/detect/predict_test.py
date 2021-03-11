"""Test cases for detection engine module."""
import unittest

from systm.detect.predict import predict_func
from systm.unittest.util import DetectTest


class TestPredict(DetectTest):
    """Test cases for systm detection prediction."""

    def test_predict(self) -> None:
        """Testcase for predict function."""
        if self.det2cfg is not None and self.cfg is not None:
            results = predict_func(self.det2cfg, self.cfg)
        else:
            self.assertEqual(True, False, msg="failed to initialize configs!")

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


if __name__ == "__main__":
    unittest.main()
