"""EMA Tests."""
import torch
from torch import nn

from vis4d.model.adapter import ModelEMAAdapter, ModelExpEMAAdapter


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
    """Test ModelEMAAdapter."""
    model = MockModel()
    model_ = ModelEMAAdapter(model, decay=0.99)
    assert model_.model.param.data == model.param.data
    assert model_.ema_model.param.data == model.param.data

    model_.model.param.data.add_(1)
    model_.update()
    out = model_.ema_model(torch.ones(1))
    assert torch.isclose(model_.ema_model.param.data, torch.tensor([1.0100]))
    assert torch.isclose(out, torch.ones(1) * 1.0100)

    model_.set(model)
    assert model_.ema_model.param.data == model.param.data


def test_model_exp_ema() -> None:
    """Test ModelExpEMAAdapter."""
    model = MockModel()
    model_ = ModelExpEMAAdapter(model, decay=0.99)
    assert model_.model.param.data == model.param.data
    assert model_.ema_model.param.data == model.param.data

    model_.model.param.data.add_(1)
    model_.update()
    out = model_.ema_model(torch.ones(1))
    assert torch.isclose(model_.ema_model.param.data, torch.tensor([1.9995]))
    assert torch.isclose(out, torch.ones(1) * 1.9995)

    model_.set(model)
    assert model_.ema_model.param.data == model.param.data
