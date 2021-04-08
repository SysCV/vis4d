"""Test cases for tracking engine prediction."""

from detectron2.data import DatasetCatalog

from openmt.track.predict import predict, track_predict_func
from openmt.unittest.util import TrackTest


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
        for ds in self.det2cfg.DATASETS.TRAIN:
            DatasetCatalog.remove(ds)
        for ds in self.det2cfg.DATASETS.TEST:
            DatasetCatalog.remove(ds)
        predict(self.cfg)
