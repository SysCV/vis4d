# type: ignore
"""Test cases for Vis4D engine Trainer."""

import os
import shutil
import unittest
from argparse import Namespace

import torch

from vis4d import config
from vis4d.engine.trainer import predict
from vis4d.engine.trainer import test as evaluate
from vis4d.engine.trainer import train
from vis4d.unittest.utils import get_test_file


class BaseEngineTests:
    """Base class for engine tests."""

    class TestDetect(unittest.TestCase):
        """Test cases for vis4d models."""

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

    class TestTrack(TestDetect):
        """Test cases for vis4d tracking."""

        predict_dir = (
            "vis4d/engine/testcases/track/bdd100k-samples/images/"
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


class TestTrackDeepSort(BaseEngineTests.TestTrack):
    """DeepSort tracking test cases."""

    predict_dir = "vis4d/engine/testcases/track/bdd100k-samples/images/"

    @classmethod
    def setUpClass(cls) -> None:
        """Set up class."""
        cls.work_dir = "./unittests/unittest_track_deepsort/"
        cls.args = Namespace(
            config=get_test_file("track/deepsort_mmdet.toml"),
            work_dir=cls.work_dir,
        )
        cls.cfg = config.parse_config(cls.args)
        if os.path.exists(
            cls.cfg.train[0].annotations.rstrip("/") + ".pkl"
        ):  # pragma: no cover
            os.remove(cls.cfg.train[0].annotations.rstrip("/") + ".pkl")


class TestTrackSort(BaseEngineTests.TestTrack):
    """Sort tracking test cases."""

    predict_dir = "vis4d/engine/testcases/track/bdd100k-samples/images/"

    def test_train(self) -> None:
        """Testcase for train."""
        # TODO add an exception that says "No Re-ID model specified" if there
        #  is no re-id model
        #  (i.e. the user wants to execute sort instead of deepsort)
        self.assertIsNotNone(self.cfg)
        self.cfg.launch.action = "train"
        self.cfg.launch.seed = 42
        trainer_args = {}
        if torch.cuda.is_available():
            trainer_args["gpus"] = "0,"  # pragma: no cover
        self.assertRaises(AttributeError, train, (self.cfg, trainer_args))
        self.cfg.launch.seed = -1

    @classmethod
    def setUpClass(cls) -> None:
        """Set up class."""
        cls.work_dir = "./unittests/unittest_track_sort/"
        cls.args = Namespace(
            config=get_test_file("track/sort_mmdet.toml"),
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
            "vis4d/engine/testcases/track/kitti-samples/"
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


class TestInsSegD2(BaseEngineTests.TestDetect):
    """Detectron2 instance segmentation test cases."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up class."""
        cls.work_dir = "./unittests/unittest_ins_seg_d2/"
        cls.args = Namespace(
            config=get_test_file("detect/mask_rcnn_d2.toml"),
            work_dir=cls.work_dir,
        )
        cls.cfg = config.parse_config(cls.args)


class TestInsSegMM(BaseEngineTests.TestDetect):
    """MMDetection instance segmentation test cases."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up class."""
        cls.work_dir = "./unittests/unittest_ins_seg_mm/"
        args = Namespace(
            config=get_test_file("detect/mask_rcnn_mmdet.toml"),
            work_dir=cls.work_dir,
        )
        cls.cfg = config.parse_config(args)


class TestSegTrackD2(BaseEngineTests.TestTrack):
    """Detectron2 segmentation tracking test cases."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up class."""
        cls.work_dir = "./unittests/unittest_seg_track_d2/"
        args = Namespace(
            config=get_test_file("track/mask_qdtrack_d2.toml"),
            work_dir=cls.work_dir,
        )
        cls.cfg = config.parse_config(args)


class TestSegTrackMM(BaseEngineTests.TestTrack):
    """MMDetection segmentation tracking test cases."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up class."""
        cls.work_dir = "./unittests/unittest_seg_track_mm/"
        args = Namespace(
            config=get_test_file("track/mask_qdtrack_mmdet.toml"),
            work_dir=cls.work_dir,
        )
        cls.cfg = config.parse_config(args)


class TestSemSegMM(BaseEngineTests.TestDetect):
    """MMSegmenation semantic segmentation test cases."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up class."""
        cls.work_dir = "./unittests/unittest_sem_seg_mm/"
        args = Namespace(
            config=get_test_file("segment/deeplabv3plus_mmseg.toml"),
            work_dir=cls.work_dir,
        )
        cls.cfg = config.parse_config(args)
