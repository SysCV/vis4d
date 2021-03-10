"""Test cases for detection engine module."""
import os
import sys
import unittest
from argparse import Namespace

from systm import config, detect, util
from systm.config.config import Dataset
from systm.detect.predict import predict_func


class TestEngine(unittest.TestCase):
    """Test cases for systm detection engine."""

    cfg = None
    args = None

    @classmethod
    def setUpClass(cls) -> None:
        """Init test cases for systm detection engine."""
        # get config
        conf = config.read_config(
            "./configs/BDD100K-Detection/retinanet_R_50_FPN.toml"
        )

        # create sample train/test datasets
        sample_dict = dict(
            name="sample_train",
            type="coco",
            data_root="./demo/bdd100k_sample",
            annotation_file="./demo/bdd100k_sample/sample_annotation.json",
        )
        conf.train = [Dataset(**sample_dict)]
        sample_dict["name"] = "sample_val"
        conf.test = [Dataset(**sample_dict)]

        # convert config to detectron2
        cfg = config.to_detectron2(conf)

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

    def test_launch_train(self) -> None:
        """Testcase for training launcher."""
        if self.args is not None and self.cfg is not None:
            detect.train(self.args, self.cfg)
        else:
            self.assertEqual(
                True, False, msg="setUpClass has failed to initialize"
            )

    def test_launch_predict(self) -> None:
        """Testcase for prediction launcher."""
        if self.args is not None and self.cfg is not None:
            detect.predict(self.args, self.cfg)
        else:
            self.assertEqual(
                True, False, msg="setUpClass has failed to initialize"
            )

    def test_predict(self) -> None:
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
