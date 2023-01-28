"""Test cases for pretty printing."""
import unittest
from typing import Any

import torch

from vis4d.common.prettyprint import PrettyRepMixin, describe_shape


class TestPrettyPrints(unittest.TestCase):
    """Test cases for pretty print functions."""

    def test_describe_shape(self) -> None:
        """Test the describe_shape function."""
        # Test dictionary input
        obj: Any = {"a": torch.rand(2, 3)}  # type: ignore
        shape_str = describe_shape(obj)
        self.assertEqual(shape_str, "{a: shape[2, 3]}")

        # Test list input
        obj = [torch.rand(2, 3), {"a": torch.rand(4, 5)}]
        shape_str = describe_shape(obj)
        self.assertEqual(shape_str, "[shape[2, 3], {a: shape[4, 5]}]")

        # Test nested dictionary and list input
        obj = {"a": [torch.rand(2, 3), torch.rand(4, 5)]}
        shape_str = describe_shape(obj)
        self.assertEqual(shape_str, "{a: [shape[2, 3], shape[4, 5]]}")

        # Test float input
        obj = 3.14
        shape_str = describe_shape(obj)
        self.assertEqual(shape_str, "3.1400")

        # Test other input type
        obj = "hello"
        shape_str = describe_shape(obj)
        self.assertEqual(shape_str, "hello")

    def test_pretty_rep_mixin(self) -> None:
        """Test the PrettyRepMixin class."""

        class TestClass(PrettyRepMixin):
            """Dummy Class."""

            def __init__(self, a: int, b: str):  # pylint: disable=invalid-name
                """Dummy init function."""
                self.a = a  # pylint: disable=invalid-name
                self.b = b  # pylint: disable=invalid-name

        obj = TestClass(1, "hello")
        self.assertEqual(str(obj), "TestClass(a=1, b=hello)")
