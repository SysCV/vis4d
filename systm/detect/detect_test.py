"""Test cases for detection module."""
import unittest
import os, sys
from argparse import Namespace
from systm import config
from systm import detect
from systm import util


class TestDetect(unittest.TestCase):
    """Test cases for systm detection."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # get config
        cfg = config.read_config(
            './configs/BDD100K-Detection/retinanet_R_50_FPN.toml')
        cfg = config.to_detectron2(cfg)

        # set device to cpu, run only 1 test iterations, batch size 1
        cfg.MODEL.DEVICE = 'cpu'
        cfg.SOLVER.MAX_ITER = 1
        cfg.SOLVER.IMS_PER_BATCH = 1
        cfg.DATASETS.TEST = ()

        # get default args, setup
        port = 2 ** 15 + 2 ** 14 + \
           hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
        args = Namespace(num_gpus=0, num_machines=1, machine_rank=0,
                         dist_url="tcp://127.0.0.1:{}".format(port),
                         resume=False)
        util.default_setup(cfg, args)

        self.cfg = cfg
        self.args = args

    def test_modules(self):
        detect.launch_module(detect.train, self.args, self.cfg)
        detect.launch_module(detect.predict, self.args, self.cfg)


if __name__ == '__main__':
    unittest.main()
