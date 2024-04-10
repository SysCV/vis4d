"""Test engine util."""

from collections import namedtuple
from dataclasses import dataclass

from vis4d.engine.util import apply_to_collection


@dataclass
class Test:
    """Test dataclass."""

    aaa: int
    bbb: int


def test_apply_to_collection():
    """Test apply_to_collection."""
    data = {"a": 1, "b": 2, "c": 3}
    data = apply_to_collection(data, int, lambda x: x * 2)
    assert data == {"a": 2, "b": 4, "c": 6}

    data = {"a": 1, "b": 2, "c": 3}
    data = apply_to_collection(data, (int, str), lambda x: x * 2)
    assert data == {"a": 2, "b": 4, "c": 6}

    data = {"a": 1, "b": 2, "c": 3}
    data = apply_to_collection(data, int, lambda x: x * 2, wrong_dtype=str)
    assert data == {"a": 2, "b": 4, "c": 6}

    data = {"a": 1, "b": 2, "c": 3}
    data = apply_to_collection(
        data, int, lambda x: x * 2, wrong_dtype=str, include_none=False
    )
    assert data == {"a": 2, "b": 4, "c": 6}

    data = {"a": 1, "b": 2, "c": 3}
    data = apply_to_collection(
        data, int, lambda x: x * 2, wrong_dtype=(str, int), include_none=False
    )
    assert data == {"a": 1, "b": 2, "c": 3}

    data = {"a": 1, "b": 2, "c": 3}
    data = apply_to_collection(
        data, int, lambda x: x * 2, wrong_dtype=(str, int), include_none=True
    )
    assert data == {"a": 1, "b": 2, "c": 3}

    # test with data as namedtuple or dataclass
    data_cls = Test(1, 2)
    data_cls = apply_to_collection(data_cls, int, lambda x: x * 2)
    assert data_cls == Test(2, 4)

    data_cls = Test(1, 2)
    data_cls = apply_to_collection(data_cls, (int, str), lambda x: x * 2)
    assert data_cls == Test(2, 4)

    data_cls = Test(1, 2)
    data_cls = apply_to_collection(
        data_cls, int, lambda x: x * 2, wrong_dtype=str
    )
    assert data_cls == Test(2, 4)

    data_cls = Test(1, 2)
    data_cls = apply_to_collection(
        data_cls,
        int,
        lambda x: x * 2,
        wrong_dtype=(str, int),
        include_none=False,
    )
    assert data_cls == Test(1, 2)

    data_cls = Test(1, 2)
    data_cls = apply_to_collection(
        data_cls,
        int,
        lambda x: x * 2,
        wrong_dtype=(str, int),
        include_none=True,
    )
    assert data_cls == Test(1, 2)

    data_tup = namedtuple("test", "aaa bbb")(1, 2)
    data_tup = apply_to_collection(data_tup, int, lambda x: x * 2)
    assert data_tup == namedtuple("test", "aaa bbb")(2, 4)

    data_tup = namedtuple("test", "aaa bbb")(1, 2)
    data_tup = apply_to_collection(data_tup, (int, str), lambda x: x * 2)
    assert data_tup == namedtuple("test", "aaa bbb")(2, 4)

    data_tup = namedtuple("test", "aaa bbb")(1, 2)
    data_tup = apply_to_collection(
        data_tup, int, lambda x: x * 2, wrong_dtype=str
    )
    assert data_tup == namedtuple("test", "aaa bbb")(2, 4)

    data_tup = namedtuple("test", "aaa bbb")(1, 2)
    data_tup = apply_to_collection(
        data_tup,
        int,
        lambda x: x * 2,
        wrong_dtype=(str, int),
        include_none=False,
    )
    assert data_tup == namedtuple("test", "aaa bbb")(1, 2)

    data_tup = namedtuple("test", "aaa bbb")(1, 2)
    data_tup = apply_to_collection(
        data_tup,
        int,
        lambda x: x * 2,
        wrong_dtype=(str, int),
        include_none=True,
    )
    assert data_tup == namedtuple("test", "aaa bbb")(1, 2)
