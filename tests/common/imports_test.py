"""Test imports."""
import unittest

from vis4d.common.imports import package_available


class TestImports(unittest.TestCase):
    """Test cases for imports."""

    def test_package_available(self) -> None:
        """Tests the package_available function."""
        self.assertTrue(package_available("numpy"))
        self.assertFalse(package_available("nonexistent_package"))
