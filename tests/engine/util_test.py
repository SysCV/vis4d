"""Test engine util."""
from collections import namedtuple
from dataclasses import dataclass

import torch
from torch import nn

from vis4d.engine.util import ModelEMA, apply_to_collection


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
    data = apply_to_collection(data_cls, int, lambda x: x * 2)
    assert data == Test(2, 4)

    data_cls = Test(1, 2)
    data = apply_to_collection(data_cls, (int, str), lambda x: x * 2)
    assert data == Test(2, 4)

    data_cls = Test(1, 2)
    data = apply_to_collection(data_cls, int, lambda x: x * 2, wrong_dtype=str)
    assert data == Test(2, 4)

    data_cls = Test(1, 2)
    data = apply_to_collection(
        data_cls,
        int,
        lambda x: x * 2,
        wrong_dtype=(str, int),
        include_none=False,
    )
    assert data == Test(1, 2)

    data_cls = Test(1, 2)
    data = apply_to_collection(
        data_cls,
        int,
        lambda x: x * 2,
        wrong_dtype=(str, int),
        include_none=True,
    )
    assert data == Test(1, 2)

    data_tup = namedtuple("test", "aaa bbb")(1, 2)
    data = apply_to_collection(data_tup, int, lambda x: x * 2)
    assert data == namedtuple("test", "aaa bbb")(2, 4)

    data_tup = namedtuple("test", "aaa bbb")(1, 2)
    data = apply_to_collection(data_tup, (int, str), lambda x: x * 2)
    assert data == namedtuple("test", "aaa bbb")(2, 4)

    data_tup = namedtuple("test", "aaa bbb")(1, 2)
    data = apply_to_collection(data_tup, int, lambda x: x * 2, wrong_dtype=str)
    assert data == namedtuple("test", "aaa bbb")(2, 4)

    data_tup = namedtuple("test", "aaa bbb")(1, 2)
    data = apply_to_collection(
        data_tup,
        int,
        lambda x: x * 2,
        wrong_dtype=(str, int),
        include_none=False,
    )
    assert data == namedtuple("test", "aaa bbb")(1, 2)

    data_tup = namedtuple("test", "aaa bbb")(1, 2)
    data = apply_to_collection(
        data_tup,
        int,
        lambda x: x * 2,
        wrong_dtype=(str, int),
        include_none=True,
    )
    assert data == namedtuple("test", "aaa bbb")(1, 2)


class MockModel(nn.Module):
    """Mock model."""

    def __init__(self) -> None:
        """Init."""
        super().__init__()
        self.param = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        return x * self.param


def test_model_ema() -> None:
    """Test ModelEMA."""
    model = MockModel()
    ema_model = ModelEMA(model, decay=0.99)
    assert ema_model.module.param.data == model.param.data

    model.param.data.add_(1)
    ema_model.update(model)
    out = ema_model(torch.ones(1))
    assert ema_model.module.param.data == torch.tensor([1.0100])
    assert out == torch.ones(1) * 1.0100

    ema_model.set(model)
    assert ema_model.module.param.data == model.param.data
