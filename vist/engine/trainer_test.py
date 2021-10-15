# type: ignore
"""Test cases for VisT engine Trainer."""

import os
import shutil
import unittest
from argparse import Namespace

import torch

from vist import config
from vist.engine.trainer import predict
from vist.engine.trainer import test as evaluate
from vist.engine.trainer import train
from vist.unittest.utils import get_test_file


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
            trainer_args = {}
            if torch.cuda.is_available():
                trainer_args["gpus"] = "0,"  # pragma: no cover
            self.cfg.launch.visualize = True
            predict(self.cfg, trainer_args)

        def test_train(self) -> None:
            """Testcase for training."""
            self.assertIsNotNone(self.cfg)
            self.cfg.launch.action = "train"
            self.cfg.launch.seed = 42
            trainer_args = {}
            if torch.cuda.is_available():
                trainer_args["gpus"] = "0,"  # pragma: no cover
            train(self.cfg, trainer_args)
            self.cfg.launch.seed = -1

        def test_testfunc(self) -> None:
            """Testcase for test function."""
            self.assertIsNotNone(self.cfg)
            self.cfg.launch.action = "test"
            trainer_args = {}
            if torch.cuda.is_available():
                trainer_args["gpus"] = "0,"  # pragma: no cover
            evaluate(self.cfg, trainer_args)

        @classmethod
        def tearDownClass(cls) -> None:
            """Clean up dataset registry, files."""
            shutil.rmtree(cls.work_dir, ignore_errors=True)

    class TestTrack(unittest.TestCase):
        """Test cases for vist tracking."""

        cfg = None
        work_dir = None
        args = None
        predict_dir = (
            "vist/engine/testcases/track/bdd100k-samples/images/"
            "00091078-875c1f73/"
        )

        def test_predict(self) -> None:
            """Testcase for predict."""
            self.assertIsNotNone(self.cfg)
            self.cfg.launch.action = "predict"
            trainer_args = {}
            if torch.cuda.is_available():
                trainer_args["gpus"] = "0,"  # pragma: no cover
            self.cfg.launch.input_dir = self.predict_dir
            self.cfg.launch.visualize = True
            predict(self.cfg, trainer_args)

        def test_testfunc(self) -> None:
            """Testcase for test function."""
            self.assertIsNotNone(self.cfg)
            self.cfg.launch.action = "test"
            trainer_args = {}
            if torch.cuda.is_available():
                trainer_args["gpus"] = "0,"  # pragma: no cover
            evaluate(self.cfg, trainer_args)

        def test_train(self) -> None:
            """Testcase for training."""
            self.assertIsNotNone(self.cfg)
            self.cfg.launch.action = "train"
            self.cfg.launch.seed = 42
            trainer_args = {}
            if torch.cuda.is_available():
                trainer_args["gpus"] = "0,"  # pragma: no cover
            train(self.cfg, trainer_args)
            self.cfg.launch.seed = -1

        @classmethod
        def tearDownClass(cls) -> None:
            """Clean up dataset registry, files."""
            shutil.rmtree(cls.work_dir, ignore_errors=True)


class TestTrackD2(BaseEngineTests.TestTrack):
    """Detectron2 tracking test cases."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up class."""
        cls.work_dir = "./unittests/unittest_track_d2/"
        cls.args = Namespace(
            config=get_test_file("track/qdtrack_d2.toml"),
            work_dir=cls.work_dir,
        )
        cls.cfg = config.parse_config(cls.args)
        if os.path.exists(
            cls.cfg.train[0].annotations.rstrip("/") + ".pkl"
        ):  # pragma: no cover
            os.remove(cls.cfg.train[0].annotations.rstrip("/") + ".pkl")


class TestTrackMM(BaseEngineTests.TestTrack):
    """MMDetection tracking test cases."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up class."""
        cls.work_dir = "./unittests/unittest_track_mm/"
        cls.args = Namespace(
            config=get_test_file("track/qdtrack_mmdet.toml"),
            work_dir=cls.work_dir,
        )
        cls.cfg = config.parse_config(cls.args)
        if os.path.exists(
            cls.cfg.train[0].annotations.rstrip("/") + ".pkl"
        ):  # pragma: no cover
            os.remove(cls.cfg.train[0].annotations.rstrip("/") + ".pkl")


class TestTrack3D(BaseEngineTests.TestTrack):
    """3D tracking test cases."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up class."""
        cls.work_dir = "./unittests/unittest_track_3d/"
        cls.args = Namespace(
            config=get_test_file("track/qd3dt_kitti.toml"),
            work_dir=cls.work_dir,
        )
        cls.cfg = config.parse_config(cls.args)
        if os.path.exists(
            cls.cfg.train[0].annotations.rstrip("/") + ".pkl"
        ):  # pragma: no cover
            os.remove(cls.cfg.train[0].annotations.rstrip("/") + ".pkl")


class TestTrackMMKITTI(BaseEngineTests.TestTrack):
    """MMDetection tracking test cases on KITTI."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up class."""
        cls.work_dir = "./unittests/unittest_track_mm_kitti/"
        cls.args = Namespace(
            config=get_test_file("track/qdtrack_mmdet_kitti.toml"),
            work_dir=cls.work_dir,
        )
        cls.cfg = config.parse_config(cls.args)
        cls.cfg.test = [cls.cfg.test[0]]  # remove multi-sensor kitti dataset
        cls.predict_dir = (
            "vist/engine/testcases/track/kitti-samples/"
            "tracking/training/image_02/0001/"
        )


class TestDetectD2(BaseEngineTests.TestDetect):
    """Detectron2 detection test cases."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up class."""
        cls.work_dir = "./unittests/unittest_detect_d2/"
        cls.args = Namespace(
            config=get_test_file("detect/faster_rcnn_d2.toml"),
            work_dir=cls.work_dir,
        )
        cls.cfg = config.parse_config(cls.args)


class TestDetectMM(BaseEngineTests.TestDetect):
    """MMDetection detection test cases."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up class."""
        cls.work_dir = "./unittests/unittest_detect_mm/"
        args = Namespace(
            config=get_test_file("detect/faster_rcnn_mmdet.toml"),
            work_dir=cls.work_dir,
        )
        cls.cfg = config.parse_config(args)
