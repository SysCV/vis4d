"""MLP Layers."""
from __future__ import annotations

import torch
from torch import nn


class ResnetBlockFC(nn.Module):
    """Fully connected ResNet Block consisting of two linear layers."""

    def __init__(
        self,
        size_in: int,
        size_out: int | None = None,
        size_h: int | None = None,
    ):
        """Fully connected ResNet Block consisting of two linear layers.

        Args:
            size_in: (int) input dimension
            size_out: Optional(int) output dimension,
                                    if not specified same as input
            size_h: Optional(int) hidden dimension,
                                  if not specfied same as min(in,out)
        """
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Applies the layer.

        Args:
            data: (tensor) input shape [N, C]
        """
        return self._call_impl(data)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Applies the layer.

        Args:
            data: (tensor) input shape [N, C]
        """
        net = self.fc_0(self.actvn(data))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(data)
        else:
            x_s = data

        return x_s + dx
