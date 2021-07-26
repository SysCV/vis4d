# type: ignore
"""Test cases for VisT engine Trainer."""

import shutil
import unittest
from argparse import Namespace

import torch

from vist import config
from vist.engine.trainer import predict
from vist.engine.trainer import test as evaluate
from vist.engine.trainer import train
from vist.unittest.utils import d2_data_reset, get_test_file

from .utils import _register


class BaseEngineTests:
    """Base class for engine tests."""

    class TestDetect(unittest.TestCase):
        """Test cases for vist models."""

        cfg = None
        work_dir = None
        args = None

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
                "AP_pedestrian",
                "AP_rider",
                "AP_car",
                "AP_truck",
                "AP_bus",
                "AP_train",
                "AP_motorcycle",
                "AP_bicycle",
                "AP_traffic light",
                "AP_traffic sign",
            ]

            for k in results:
                self.assertIn(k, metric_keys)

        @classmethod
        def tearDownClass(cls) -> None:
            """Clean up dataset registry, files."""
            assert cls.cfg.train is not None, "cfg.train must not be None"
            assert cls.cfg.test is not None, "cfg.test must not be None"
            d2_data_reset([ds.name for ds in cls.cfg.train])
            d2_data_reset([ds.name for ds in cls.cfg.test])
            d2_data_reset(["00091078-875c1f73"])
            shutil.rmtree(cls.work_dir, ignore_errors=True)

    class TestTrack(unittest.TestCase):
        """Test cases for vist tracking."""

        cfg = None
        work_dir = None
        args = None

        def test_predict(self) -> None:
            """Testcase for predict."""
            self.assertIsNotNone(self.cfg)
            self.cfg.launch.action = "predict"
            if torch.cuda.is_available():
                self.cfg.launch.device = "cuda"  # pragma: no cover
            self.cfg.launch.input_dir = (
                "vist/engine/testcases/track/"
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
            self.assertIsNotNone(self.cfg.train)
            self.assertIsNotNone(self.cfg.test)
            _register(self.cfg.train)
            _register(self.cfg.test)
            _register(self.cfg.train)

        @classmethod
        def tearDownClass(cls) -> None:
            """Clean up dataset registry, files."""
            assert cls.cfg.train is not None, "cfg.train must not be None"
            assert cls.cfg.test is not None, "cfg.test must not be None"
            d2_data_reset([ds.name for ds in cls.cfg.train])
            d2_data_reset([ds.name for ds in cls.cfg.test])
            d2_data_reset(["00091078-875c1f73"])
            shutil.rmtree(cls.work_dir, ignore_errors=True)


class TestTrackD2(BaseEngineTests.TestTrack):
    """Detectron2 tracking test cases."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up class."""
        cls.work_dir = "./unittest_track_d2/"
        cls.args = Namespace(
            config=get_test_file("track/qdtrack_d2.toml"),
            output_dir=cls.work_dir,
        )
        cls.cfg = config.parse_config(cls.args)


class TestTrackMM(BaseEngineTests.TestTrack):
    """MMDetection tracking test cases."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up class."""
        cls.work_dir = "./unittest_track_mm/"
        cls.args = Namespace(
            config=get_test_file("track/qdtrack_mmdet.toml"),
            output_dir=cls.work_dir,
        )
        cls.cfg = config.parse_config(cls.args)


class TestDetectD2(BaseEngineTests.TestDetect):
    """Detectron2 detection test cases."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up class."""
        cls.work_dir = "./unittest_detect_d2/"
        cls.args = Namespace(
            config=get_test_file("detect/faster_rcnn_d2.toml"),
            output_dir=cls.work_dir,
        )
        cls.cfg = config.parse_config(cls.args)


class TestDetectMM(BaseEngineTests.TestDetect):
    """MMDetection detection test cases."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up class."""
        cls.work_dir = "./unittest_detect_mm/"
        args = Namespace(
            config=get_test_file("detect/faster_rcnn_mmdet.toml"),
            output_dir=cls.work_dir,
        )
        cls.cfg = config.parse_config(args)
