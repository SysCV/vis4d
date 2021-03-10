"""Test cases for detection engine module."""
import os
import sys
import unittest
from argparse import Namespace

from systm import config, detect, util
from systm.detect.predict import predict_func


class TestEngine(unittest.TestCase):
    """Test cases for systm detection engine."""

    @classmethod
    def setUpClass(cls):
        """Init test cases for systm detection engine."""
        # get config
        cfg = config.read_config(
            "./configs/BDD100K-Detection/retinanet_R_50_FPN.toml"
        )

        # modify train/test datasets to demo/bdd100k_sample
        cfg.train[0].data_root = cfg.test[
            0
        ].data_root = "./demo/bdd100k_sample"
        cfg.train[0].annotation_file = cfg.test[
            0
        ].annotation_file = "./demo/bdd100k_sample/sample_annotation.json"

        # convert config to detectron2
        cfg = config.to_detectron2(cfg)

        # set device to cpu, run only 1 test iterations, batch size 1
        cfg.MODEL.DEVICE = "cpu"
        cfg.SOLVER.MAX_ITER = 1
        cfg.SOLVER.IMS_PER_BATCH = 1

        # get default args, setup
        port = (
            2 ** 15
            + 2 ** 14
            + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
        )
        args = Namespace(
            num_gpus=0,
            num_machines=1,
            machine_rank=0,
            dist_url="tcp://127.0.0.1:{}".format(port),
            resume=False,
        )
        util.default_setup(cfg, args)

        cls.cfg = cfg
        cls.args = args

    def test_launch_train(self):
        """Testcase for training launcher."""
        detect.train(self.args, self.cfg)

    def test_launch_predict(self):
        """Testcase for prediction launcher."""
        detect.predict(self.args, self.cfg)

    def test_predict(self):
        """Testcase for predict function."""
        results = predict_func(self.cfg, False)

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
