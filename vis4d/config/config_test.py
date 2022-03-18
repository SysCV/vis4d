"""Test cases for parsing config."""
import unittest
from argparse import Namespace

from ..unittest.utils import get_test_file
from .config import parse_config, read_config


class TestLoadConfig(unittest.TestCase):
    """Test cases for vis4d config parsing."""

    def test_det_yaml(self) -> None:
        """Check models configuration in yaml format."""
        config = read_config(get_test_file("config-det.yaml"))
        self.assertEqual(config.model["type"], "test-model")
        self.assertEqual(config.launch.samples_per_gpu, 2)

    def test_det_toml(self) -> None:
        """Check models configuration in toml format."""
        config = read_config(get_test_file("config-det.toml"))
        self.assertEqual(config.model["type"], "test-model")
        self.assertEqual(config.launch.samples_per_gpu, 2)

    def test_det_args(self) -> None:
        """Check cmd line argument parsing to launch cfg."""
        args = Namespace(
            config=get_test_file("config-det.yaml"),
            device="cuda",
            samples_per_gpu=2,
            cfg_options="model.image_channel_mode=BGR",
        )
        cfg = parse_config(args)
        self.assertEqual(cfg.launch.samples_per_gpu, 2)
        self.assertEqual(cfg.model["image_channel_mode"], "BGR")

    def test_det_notsupported(self) -> None:
        """Check models configuration in not-supported format."""
        self.assertRaises(NotImplementedError, read_config, "")

    def test_list_replacemnet(self) -> None:
        """Check cmd line argument parsing to launch cfg."""
        args = Namespace(
            config=get_test_file("config-det.toml"),
            cfg_options="train.1.name=trainer-temp",
        )
        cfg = parse_config(args)
        self.assertEqual(cfg.train[1]["name"], "trainer-temp")
