"""Test the dict module."""

import unittest

from vis4d.common.dict import get_dict_nested, set_dict_nested


class TestDictUtils(unittest.TestCase):
    """Test cases for array conversion ops."""

    def test_set_dict_nested(self) -> None:
        """Tests the set_dict_nested function."""
        d = {}  # type:ignore
        set_dict_nested(d, ["a", "b", "c"], 10)
        self.assertEqual(d, {"a": {"b": {"c": 10}}})

        d = {"a": {"b": {"c": 10}}}
        set_dict_nested(d, ["a", "b", "c"], 20)
        self.assertEqual(d, {"a": {"b": {"c": 20}}})

        d = {"a": {"b": {"c": 10}}}
        set_dict_nested(d, ["a", "b", "d"], 20)
        self.assertEqual(d, {"a": {"b": {"c": 10, "d": 20}}})

        d = {"a": {"b": {"c": 10}}}
        set_dict_nested(d, ["a", "e", "f"], 20)
        self.assertEqual(d, {"a": {"b": {"c": 10}, "e": {"f": 20}}})

    def test_get_dict_nested(self) -> None:
        """Tests the get_dict_nested function."""
        d = {"a": {"b": {"c": 10}}}
        self.assertEqual(get_dict_nested(d, ["a", "b", "c"]), 10)
        assert get_dict_nested(d, ["a", "b", "c"]) == 10

        d = {"a": {"b": {"c": 10, "d": 20}}}
        self.assertEqual(get_dict_nested(d, ["a", "b", "c"]), 10)
        self.assertEqual(get_dict_nested(d, ["a", "b", "d"]), 20)
