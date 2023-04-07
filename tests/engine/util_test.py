"""Test engine util."""
from collections import namedtuple
from dataclasses import dataclass

from vis4d.engine.util import apply_to_collection


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


def test_apply_to_collection_dataclass():
    """Test apply_to_collection with data as namedtuple or dataclass."""

    @dataclass
    class Test:
        a: int
        b: int

    data = Test(1, 2)
    data = apply_to_collection(data, int, lambda x: x * 2)
    assert data == Test(2, 4)

    data = Test(1, 2)
    data = apply_to_collection(data, (int, str), lambda x: x * 2)
    assert data == Test(2, 4)

    data = Test(1, 2)
    data = apply_to_collection(data, int, lambda x: x * 2, wrong_dtype=str)
    assert data == Test(2, 4)

    data = Test(1, 2)
    data = apply_to_collection(
        data, int, lambda x: x * 2, wrong_dtype=(str, int), include_none=False
    )
    assert data == Test(1, 2)

    data = Test(1, 2)
    data = apply_to_collection(
        data, int, lambda x: x * 2, wrong_dtype=(str, int), include_none=True
    )
    assert data == Test(1, 2)

    data = namedtuple("test", "a b")(1, 2)
    data = apply_to_collection(data, int, lambda x: x * 2)
    assert data == namedtuple("test", "a b")(2, 4)

    data = namedtuple("test", "a b")(1, 2)
    data = apply_to_collection(data, (int, str), lambda x: x * 2)
    assert data == namedtuple("test", "a b")(2, 4)

    data = namedtuple("test", "a b")(1, 2)
    data = apply_to_collection(data, int, lambda x: x * 2, wrong_dtype=str)
    assert data == namedtuple("test", "a b")(2, 4)

    data = namedtuple("test", "a b")(1, 2)
    data = apply_to_collection(
        data, int, lambda x: x * 2, wrong_dtype=(str, int), include_none=False
    )
    assert data == namedtuple("test", "a b")(1, 2)

    data = namedtuple("test", "a b")(1, 2)
    data = apply_to_collection(
        data, int, lambda x: x * 2, wrong_dtype=(str, int), include_none=True
    )
    assert data == namedtuple("test", "a b")(1, 2)
