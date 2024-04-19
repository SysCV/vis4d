"""Adapter for counting flops in a model."""

from __future__ import annotations

from typing import Any

from torch import nn

from vis4d.engine.connectors import DataConnector

# Ops to ignore from counting, including elementwise and reduction ops
IGNORED_OPS = {
    "aten::add",
    "aten::add_",
    "aten::argmax",
    "aten::argsort",
    "aten::batch_norm",
    "aten::constant_pad_nd",
    "aten::div",
    "aten::div_",
    "aten::exp",
    "aten::log2",
    "aten::max_pool2d",
    "aten::meshgrid",
    "aten::mul",
    "aten::mul_",
    "aten::neg",
    "aten::nonzero_numpy",
    "aten::reciprocal",
    "aten::repeat_interleave",
    "aten::rsub",
    "aten::sigmoid",
    "aten::sigmoid_",
    "aten::softmax",
    "aten::sort",
    "aten::sqrt",
    "aten::sub",
    "torchvision::nms",
}


class FlopsModelAdapter(nn.Module):
    """Adapter for the model to count flops."""

    def __init__(
        self, model: nn.Module, data_connector: DataConnector
    ) -> None:
        """Initialize the adapter."""
        super().__init__()
        self.model = model
        self.data_connector = data_connector

    def forward(self, *args: Any) -> Any:  # type: ignore
        """Forward pass through the model."""
        data_dict = {}
        for i, key in enumerate(self.data_connector.key_mapping):
            data_dict[key] = args[0][i]

        return self.model(**data_dict)
