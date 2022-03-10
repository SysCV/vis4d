"""Unit tests for engine utils."""
import logging
import unittest
from argparse import Namespace

from .utils import setup_logger

logger = logging.getLogger("pytorch_lightning")


class TestTrack(unittest.TestCase):
    """Test cases for vis4d tracking."""

    def test_split_args(self) -> None:
        """Test split_args function."""
        args = Namespace(a="hi", max_steps=10)
        args1, args2 = split_args(args)
        self.assertEqual(args1.a, "hi")  # pylint: disable=no-member
        self.assertEqual(list(vars(args1).keys()), ["a"])
        self.assertEqual(args.max_steps, 10)  # pylint: disable=no-member
        self.assertEqual(list(args2.keys()), ["max_steps"])

    @staticmethod
    def test_setup_logger() -> None:
        """Test setup_logger."""
        setup_logger()
        logger.debug("DEBUG")
        logger.info("INFO")
        logger.warning("WARN")
        logger.error("ERROR")
        logger.critical("CRITICAL")
        setup_logger(color=False)
        logger.debug("DEBUG")
        logger.info("INFO")
        logger.warning("WARN")
        logger.error("ERROR")
        logger.critical("CRITICAL")
