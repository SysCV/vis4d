"""Test logging."""

import logging
import os
import shutil
import tempfile
import unittest

from vis4d.common.logging import setup_logger


class TestLogging(unittest.TestCase):
    """Test cases for logging."""

    def test_setup_logger(self) -> None:
        """Test the setup_logger function."""
        logger = logging.getLogger("vis4d.common.logging")
        tmpdir = tempfile.mkdtemp()
        filepath = os.path.join(tmpdir, "test.log")
        setup_logger(
            logger, filepath=filepath, color=False, std_out_level=logging.DEBUG
        )
        logger.info("This is a test")
        logger.warning("This is a test")
        logger.error("This is a test")
        logger.critical("This is a test")

        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 4
        assert "INFO" in lines[0]
        assert "WARNING" in lines[1]
        assert "ERROR" in lines[2]
        assert "CRITICAL" in lines[3]

        setup_logger(
            logger, filepath=filepath, color=True, std_out_level=logging.DEBUG
        )
        logger.info("This is a test")
        logger.warning("This is a test")
        logger.error("This is a test")
        logger.critical("This is a test")

        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 12
        assert "INFO" in lines[4]
        assert "WARNING" in lines[6]
        assert "ERROR" in lines[8]
        assert "CRITICAL" in lines[10]

        shutil.rmtree(tmpdir)
