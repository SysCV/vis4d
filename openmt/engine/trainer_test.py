"""Test cases for openMT engine Trainer."""
import unittest
from argparse import Namespace

import torch

from openmt import config
from openmt.engine.trainer import predict
from openmt.engine.trainer import test as evaluate
from openmt.engine.trainer import train
from openmt.unittest.utils import d2_data_reset, get_test_file

from .utils import _register


class TestTrack(unittest.TestCase):
    """Test cases for openmt tracking."""

    args = Namespace(config=get_test_file("track/quasi_dense_R_50_FPN.toml"))
    cfg = config.parse_config(args)

    def test_predict(self) -> None:
        """Testcase for predict."""
        self.assertIsNotNone(self.cfg)
        self.cfg.launch.action = "predict"
        if torch.cuda.is_available():
            self.cfg.launch.device = "cuda"  # pragma: no cover
        self.cfg.launch.input_dir = (
            "openmt/engine/testcases/track/"
            "bdd100k-samples/images/00091078-875c1f73/"
        )
        self.cfg.launch.visualize = True
        predict(self.cfg)

    def test_testfunc(self) -> None:
        """Testcase for test function."""
        self.assertIsNotNone(self.cfg)
        self.cfg.launch.action = "predict"
        if torch.cuda.is_available():
            self.cfg.launch.device = "cuda"  # pragma: no cover
        results = evaluate(self.cfg)["track"]
        metric_keys = [
            "pedestrian",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
            "human",
            "vehicle",
            "bike",
            "OVERALL",
        ]
        for k in results:
            self.assertIn(k, metric_keys)

    def test_train(self) -> None:
        """Testcase for training."""
        self.assertIsNotNone(self.cfg)
        self.cfg.launch.action = "train"
        self.cfg.launch.seed = 42
        if torch.cuda.is_available():
            self.cfg.launch.device = "cuda"  # pragma: no cover
        train(self.cfg)
        self.cfg.launch.seed = -1

    def test_duplicate_register(self) -> None:
        """Test if duplicated datasets are skipped."""
        assert self.cfg.train is not None
        assert self.cfg.test is not None
        _register(self.cfg.train)
        _register(self.cfg.test)
        _register(self.cfg.train)

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up dataset registry."""
        assert cls.cfg.train is not None
        assert cls.cfg.test is not None
        d2_data_reset(cls.cfg.train)
        d2_data_reset(cls.cfg.test)


class TestDetect(unittest.TestCase):
    """Test cases for openmt detection."""

    args = Namespace(config=get_test_file("detect/faster_rcnn_R_50_FPN.toml"))
    cfg = config.parse_config(args)

    def test_predict(self) -> None:
        """Testcase for predict."""
        self.assertIsNotNone(self.cfg)
        self.cfg.launch.action = "predict"
        if torch.cuda.is_available():
            self.cfg.launch.device = "cuda"  # pragma: no cover
        self.cfg.launch.visualize = True
        predict(self.cfg)

    def test_train(self) -> None:
        """Testcase for training."""
        self.assertIsNotNone(self.cfg)
        self.cfg.launch.action = "train"
        self.cfg.launch.seed = 42
        if torch.cuda.is_available():
            self.cfg.launch.device = "cuda"  # pragma: no cover
        train(self.cfg)
        self.cfg.launch.seed = -1

    def test_testfunc(self) -> None:
        """Testcase for test function."""
        self.assertIsNotNone(self.cfg)
        self.cfg.launch.action = "predict"
        if torch.cuda.is_available():
            self.cfg.launch.device = "cuda"  # pragma: no cover
        results = evaluate(self.cfg)["detect"]
        metric_keys = [
            "AP",
            "AP_50",
            "AP_75",
            "AP_small",
            "AP_medium",
            "AP_large",
            "AR_max_1",
            "AR_max_10",
            "AR_max_100",
            "AR_small",
            "AR_medium",
            "AR_large",
        ]

        for k in results:
            self.assertIn(k, metric_keys)

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up dataset registry."""
        assert cls.cfg.train is not None
        assert cls.cfg.test is not None
        d2_data_reset(cls.cfg.train)
        d2_data_reset(cls.cfg.test)
