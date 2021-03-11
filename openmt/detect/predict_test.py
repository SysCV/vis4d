"""Test cases for detection engine prediction."""

from detectron2.data import DatasetCatalog

from openmt.detect.predict import predict, predict_func
from openmt.unittest.util import DetectTest


class TestPredict(DetectTest):
    """Test cases for openmt detection prediction."""

    def test_predict(self) -> None:
        """Testcase for predict function."""
        self.assertIsNotNone(self.det2cfg)
        self.assertIsNotNone(self.cfg)
        results = predict_func(self.det2cfg, self.cfg)

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

    def test_predict_launcher(self) -> None:
        """Testcase for prediction launcher."""
        for ds in self.det2cfg.DATASETS.TRAIN:
            DatasetCatalog.remove(ds)
        for ds in self.det2cfg.DATASETS.TEST:
            DatasetCatalog.remove(ds)
        predict(self.cfg)
