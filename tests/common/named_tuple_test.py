"""Test named tuple."""
import unittest
from collections import namedtuple

from vis4d.common.named_tuple import get_all_keys, get_from_namedtuple


class NamedTupleTest(unittest.TestCase):
    """Named tuple test class."""

    def test_get_all_keys(self):
        """Test get_all_keys."""
        Test = namedtuple("Test", ["a", "b"])
        Test2 = namedtuple("Test2", ["c", "d"])
        Test3 = namedtuple("Test3", ["e", "f"])
        Test4 = namedtuple("Test4", ["g", "h"])
        Test5 = namedtuple("Test5", ["i", "j"])
        test = Test(Test2(Test3(Test4(Test5(1, 2), 3), 4), 5), 6)
        assert get_all_keys(test) == [
            "a.c.e.g.i",
            "a.c.e.g.j",
            "a.c.e.h",
            "a.c.f",
            "a.d",
            "b",
        ]

    def test_get_from_namedtuple(self):
        """Test get_from_namedtuple."""
        Test = namedtuple("Test", ["a", "b"])
        Test2 = namedtuple("Test2", ["c", "d"])
        Test3 = namedtuple("Test3", ["e", "f"])
        Test4 = namedtuple("Test4", ["g", "h"])
        Test5 = namedtuple("Test5", ["i", "j"])
        test = Test(Test2(Test3(Test4(Test5(1, 2), 3), 4), 5), 6)
        assert get_from_namedtuple(test, "a.c.e.g.i") == 1
        assert get_from_namedtuple(test, "a.c.e.g.j") == 2
        assert get_from_namedtuple(test, "a.c.e.h") == 3
        assert get_from_namedtuple(test, "a.c.f") == 4
        assert get_from_namedtuple(test, "a.d") == 5
        assert get_from_namedtuple(test, "b") == 6
        with self.assertRaises(ValueError):
            get_from_namedtuple(test, "a.c.e.g.k")
