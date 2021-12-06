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

    class TestBase(unittest.TestCase):
        """Base test case for vis4d models."""

        cfg = None
        work_dir = None
        args = None

        @classmethod
        def tearDownClass(cls) -> None:
            """Clean up dataset registry, files."""
            shutil.rmtree(cls.work_dir, ignore_errors=True)

    class TestTrain(TestBase):
        """Base test case for vis4d models."""

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

    class TestTest(TestBase):
        """Base test case for vis4d models."""

        def test_testfunc(self) -> None:
            """Testcase for test function."""
            self.assertIsNotNone(self.cfg)
            self.cfg.launch.action = "test"
            trainer_args = {}
            if torch.cuda.is_available():
                trainer_args["gpus"] = "0,"  # pragma: no cover
            evaluate(self.cfg, trainer_args)

    class TestDetect(TestTrain, TestTest):
        """Test cases for vis4d models."""

        def test_predict(self) -> None:
            """Testcase for predict."""
            self.assertIsNotNone(self.cfg)
            self.cfg.launch.action = "predict"
            trainer_args = {}
            if torch.cuda.is_available():
                trainer_args["gpus"] = "0,"  # pragma: no cover
            self.cfg.launch.visualize = True
            predict(self.cfg, trainer_args)

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


class TestTrackD2(BaseEngineTests.TestTrain, BaseEngineTests.TestTest):
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


# class TestTrack3D(BaseEngineTests.TestTrack):
#     """3D tracking test cases."""
#
#     @classmethod
#     def setUpClass(cls) -> None:
#         """Set up class."""
#         cls.work_dir = "./unittests/unittest_track_3d/"
#         cls.args = Namespace(
#             config=get_test_file("track/qd3dt_kitti.toml"),
#             work_dir=cls.work_dir,
#         )
#         cls.cfg = config.parse_config(cls.args)
#         if os.path.exists(
#             cls.cfg.train[0].annotations.rstrip("/") + ".pkl"
#         ):  # pragma: no cover
#             os.remove(cls.cfg.train[0].annotations.rstrip("/") + ".pkl")


class TestDetectMM(BaseEngineTests.TestTest):
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


# TODO
# class TestInsSegD2(BaseEngineTests.TestDetect):
#     """Detectron2 instance segmentation test cases."""
#
#     @classmethod
#     def setUpClass(cls) -> None:
#         """Set up class."""
#         cls.work_dir = "./unittests/unittest_ins_seg_d2/"
#         cls.args = Namespace(
#             config=get_test_file("detect/mask_rcnn_d2.toml"),
#             work_dir=cls.work_dir,
#         )
#         cls.cfg = config.parse_config(cls.args)
#
#
# class TestInsSegMM(BaseEngineTests.TestTrain):
#     """MMDetection instance segmentation test cases."""
#
#     @classmethod
#     def setUpClass(cls) -> None:
#         """Set up class."""
#         cls.work_dir = "./unittests/unittest_ins_seg_mm/"
#         args = Namespace(
#             config=get_test_file("detect/mask_rcnn_mmdet.toml"),
#             work_dir=cls.work_dir,
#         )
#         cls.cfg = config.parse_config(args)
#
#
# class TestSegTrackMM(BaseEngineTests.TestTrain):
#     """MMDetection segmentation tracking test cases."""
#
#     @classmethod
#     def setUpClass(cls) -> None:
#         """Set up class."""
#         cls.work_dir = "./unittests/unittest_seg_track_mm/"
#         args = Namespace(
#             config=get_test_file("track/mask_qdtrack_mmdet.toml"),
#             work_dir=cls.work_dir,
#         )
#         cls.cfg = config.parse_config(args)
#


# class TestSemSegMM(BaseEngineTests.TestTrain):
#     """MMSegmenation semantic segmentation test cases."""
#
#     @classmethod
#     def setUpClass(cls) -> None:
#         """Set up class."""
#         cls.work_dir = "./unittests/unittest_sem_seg_mm/"
#         args = Namespace(
#             config=get_test_file("segment/deeplabv3_mmseg.toml"),
#             work_dir=cls.work_dir,
#         )
#         cls.cfg = config.parse_config(args)
#         cls.cfg.launch.tqdm = True


class TestSemSegMM2(BaseEngineTests.TestTrain):
    """MMSegmenation semantic segmentation test cases."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up class."""
        cls.work_dir = "./unittests/unittest_sem_seg_mm_2/"
        args = Namespace(
            config=get_test_file("segment/deeplabv3plus_mmseg.toml"),
            work_dir=cls.work_dir,
        )
        cls.cfg = config.parse_config(args)
        cls.cfg.launch.tqdm = True
