"""Unit tests for engine utils."""
import logging
import unittest

from .utils import setup_logger

logger = logging.getLogger("pytorch_lightning")


class TestTrack(unittest.TestCase):
    """Test cases for vis4d tracking."""

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
