"""Test the dict module."""

import unittest

from vis4d.common.dict import flatten_dict, get_dict_nested, set_dict_nested


class TestDictUtils(unittest.TestCase):
    """Test cases for array conversion ops."""

    def test_flatten_dict(self) -> None:
        """Tests the flatten_dict function."""
        d = {"a": {"b": {"c": 10}}}
        self.assertEqual(flatten_dict(d, "."), ["a.b.c"])

        d = {"a": {"b": {"c": 10, "d": 20}}}
        self.assertEqual(flatten_dict(d, "/"), ["a/b/c", "a/b/d"])

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

    def test_wrong_key(self):
        """Tests the get_dict_nested function with a wrong key."""
        d = {"a": {"b": {"c": 10}}}
        with self.assertRaises(ValueError):
            get_dict_nested(d, "e")
