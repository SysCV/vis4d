"""MLP Layers."""
from __future__ import annotations

import torch
from torch import nn, Tensor


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


class FFN(nn.Module):
    """Implements feed-forward networks (FFNs) with identity connection."""

    def __init__(
        self,
        layers: nn.Modele,
        embed_dims: int = 256,
        dropout_layer: nn.Module = nn.Identity(),
        add_identity: bool = True,
        layer_scale_init_value: float = 0.0,
    ) -> None:
        """Init FFN.

        Args:
            layers (nn.Module): The layers of the FFN.
            embed_dims (int): The feature dimension. Same as
                `MultiheadAttention`. Defaults: 256.
            dropout_layer (nn.Module): The dropout_layer used when adding the
                shortcut.
            add_identity (bool, optional): Whether to add the identity
                connection. Default: `True`.
            layer_scale_init_value (float): Initial value of scale factor in
                LayerScale. Default: 0.0
        """
        super().__init__()
        self.layers = layers

        self.dropout_layer = dropout_layer
        self.add_identity = add_identity

        if layer_scale_init_value > 0:
            self.gamma2 = LayerScale(embed_dims, scale=layer_scale_init_value)
        else:
            self.gamma2 = nn.Identity()

    def forward(self, x: Tensor, identity: Tensor | None = None) -> None:
        """Forward function for `FFN`.

        The function would add x to the output tensor if residue is None.
        """
        out = self.layers(x)
        out = self.gamma2(out)

        if self.add_identity:
            identity = x if identity is None else identity
            return identity + self.dropout_layer(out)

        return self.dropout_layer(out)


class LayerScale(nn.Module):
    """LayerScale layer."""

    def __init__(
        self,
        dim: int,
        inplace: bool = False,
        data_format: str = "channels_last",
        scale: float = 1e-5,
    ) -> None:
        """Init LayerScale.

        Args:
            dim (int): Dimension of input features.
            inplace (bool): Whether performs operation in-place. Default:
                `False`.
            data_format (str): The input data format, could be 'channels_last'
                or 'channels_first', representing (B, C, H, W) and (B, N, C)
                format data respectively. Default: 'channels_last'.
            scale (float): Initial value of scale factor. Default: 1e-5.
        """
        super().__init__()
        assert data_format in (
            "channels_last",
            "channels_first",
        ), "'data_format' could only be channels_last or channels_first."
        self.inplace = inplace
        self.data_format = data_format
        self.weight = nn.Parameter(torch.ones(dim) * scale)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function for `LayerScale`."""
        if self.data_format == "channels_first":
            shape = tuple((1, -1, *(1 for _ in range(x.dim() - 2))))
        else:
            shape = tuple((*(1 for _ in range(x.dim() - 1)), -1))

        if self.inplace:
            return x.mul_(self.weight.view(*shape))
        else:
            return x * self.weight.view(*shape)
