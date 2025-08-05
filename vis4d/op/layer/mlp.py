"""MLP Layers."""

from __future__ import annotations

from torch import Tensor, nn


class TransformerBlockMLP(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: nn.Module = nn.GELU(),
        bias: bool = True,
        drop: float = 0.0,
    ):
        """Init MLP.

        Args:
            in_features (int): Number of input features.
            hidden_features (int, optional): Number of hidden features.
                Defaults to None.
            out_features (int, optional): Number of output features.
                Defaults to None.
            act_layer (nn.Module, optional): Activation layer.
                Defaults to nn.GELU.
            bias (bool, optional): If bias should be used. Defaults to True.
            drop (float, optional): Dropout probability. Defaults to 0.0.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def __call__(self, data: Tensor) -> Tensor:
        """Applies the layer.

        Args:
            data: (tensor) input shape [N, C]
        """
        return self._call_impl(data)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: (tensor) input shape [N, C]
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
