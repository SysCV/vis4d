"""Test cases for default boilerplate logic."""
import unittest

from .defaults import default_argument_parser


class TestDefaults(unittest.TestCase):
    """Test cases for default boilerplate logic."""

    def test_argparse(self) -> None:
        """Check cmd line argument parsing."""
        parser = default_argument_parser()
        action = "train"
        cfg_path = "/path/to/config.toml"
        args = parser.parse_args([action, "--config", cfg_path])
        self.assertEqual(args.config, cfg_path)
        self.assertEqual(args.action, action)
