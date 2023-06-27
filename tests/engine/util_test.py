"""Test engine util."""
from collections import namedtuple
from dataclasses import dataclass

import torch
from torch import nn

from vis4d.config.config_dict import class_config
from vis4d.engine.util import ModelEMAAdapter, apply_to_collection


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
    model_ = ModelEMAAdapter(model, decay=0.99)
    assert model_.model.param.data == model.param.data
    assert model_.ema_model.param.data == model.param.data

    model_.model.param.data.add_(1)
    model_.update()
    out = model_.ema_model(torch.ones(1))
    assert model_.ema_model.param.data == torch.tensor([1.0100])
    assert out == torch.ones(1) * 1.0100

    model_.set(model)
    assert model_.ema_model.param.data == model.param.data

    model_cfg = class_config(MockModel)
    model_ = ModelEMAAdapter(model_cfg, decay=0.99)
    assert model_.model.param.data == torch.ones(1)
