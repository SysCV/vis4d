"""Utilities for unit tests."""
import inspect
import os
import unittest
from argparse import Namespace

from systm import config
from systm.detect import default_setup, to_detectron2


def get_test_file(file_name: str) -> str:
    """Test test file path."""
    return os.path.join(
        os.path.dirname(os.path.abspath(inspect.stack()[1][1])),
        "testcases",
        file_name,
    )


class EngineTest(unittest.TestCase):
    """Test case init for systm detection engine."""

    args = Namespace(config="systm/detect/testcases/retinanet_R_50_FPN.toml")
    cfg = config.parse_config(args)
    det2cfg = to_detectron2(cfg)
    default_setup(det2cfg, cfg.launch)
