"""Test config registry."""
from __future__ import annotations

import unittest

import pytest

from tests.util import get_test_data
from vis4d.config.util.registry import get_config_by_name, register_config


class TestRegistry(unittest.TestCase):
    """Test the config registry."""

    def test_yaml(self) -> None:
        """Test reading a yaml config file."""
        file = get_test_data(
            "config_test/bdd100k/faster_rcnn/faster_rcnn_r50_1x_bdd100k.yaml"
        )

        # Config can be resolved
        config = get_config_by_name(file)
        self.assertTrue(config is not None)

        # Config does not exist
        with pytest.raises(ValueError) as err:
            config = get_config_by_name(file.replace("r50", "r91"))
        self.assertTrue("Could not find" in str(err.value))

    def test_py(self) -> None:
        """Test reading a py config file from the model zoo."""
        file = "/bdd100k/faster_rcnn/faster_rcnn_r50_1x_bdd100k.py"
        cfg = get_config_by_name(file)
        self.assertTrue(cfg is not None)

        # Only by file name
        file = "faster_rcnn_r50_1x_bdd100k.py"
        cfg = get_config_by_name(file)
        self.assertTrue(cfg is not None)

        # Check did you mean message
        file = "faster_rcnn_r90_1x_bdd100k"
        with pytest.raises(ValueError) as err:
            cfg = get_config_by_name(file)
        self.assertTrue("faster_rcnn_r50_1x_bdd100k" in str(err.value))

    def test_zoo(self) -> None:
        """Test reading a registered config from the zoo."""
        config = get_config_by_name("faster_rcnn_r50_1x_bdd100k")
        self.assertTrue(config is not None)

        # Full Qualified Name
        config = get_config_by_name("bdd100k/faster_rcnn_r50_1x_bdd100k")
        self.assertTrue(config is not None)

        # Check did you mean message
        with pytest.raises(ValueError) as err:
            config = get_config_by_name("faster_rcnn_r90_1x_bdd100k")
        self.assertTrue("faster_rcnn_r50_1x_bdd100k" in str(err.value))

    def test_decorator(self) -> None:
        """Test registering a config."""

        @register_config("cat", "test")  # type: ignore
        def get_config() -> dict[str, str]:
            """Test config."""
            return {"test": "test"}

        config = get_config_by_name("cat/test")
        self.assertTrue(config is not None)
        self.assertEqual(config["test"], "test")
