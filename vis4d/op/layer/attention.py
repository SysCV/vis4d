import itertools

import torch
import torch.nn as nn


class Attention2d(torch.nn.Module):
    """Multi-head self-attention for 2D feature maps."""

    def __init__(
        self,
        dim: int,
        key_dim: int,
        num_heads: int = 8,
        attn_ratio: int = 4,
        resolution: tuple[int, int] = (14, 14),
    ):
        """Multi-head self-attention for 2D feature maps.

        Args:
            dim (int): Input dimension.
            key_dim (int): Key dimension.
            num_heads (int, optional): Number of attention heads. Defaults to 8.
            attn_ratio (int, optional): Attention ratio. Defaults to 4.
            resolution (tuple[int, int], optional): Resolution of the input
                feature map, (height, width). Defaults to (14, 14).
        """
        super().__init__()
        assert isinstance(resolution, tuple) and len(resolution) == 2
        self.num_heads = num_heads
        self.scale = key_dim ** (-0.5)
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.attn_dim = int(attn_ratio * key_dim)
        self.attn_dim_all_heads = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        attn_size = self.attn_dim_all_heads + nh_kd * 2

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, attn_size)
        self.proj = nn.Linear(self.attn_dim_all_heads, dim)

        # Compute attention biases and store them as a buffer for fast access.
        points = list(
            itertools.product(range(resolution[0]), range(resolution[1]))
        )
        attention_offsets: dict[tuple[int, int], int] = {}
        idxs: list[int] = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets))
        )
        self.register_buffer(
            "attention_bias_idxs",
            torch.LongTensor(idxs).view(len(points), len(points)),
            persistent=False,
        )

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, "attn_biases"):
            del self.attn_biases
        else:
            self.attn_biases = self.attention_biases[
                :, self.attention_bias_idxs
            ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x (B,N,C)
        """Forward. pass.

        Args:
            x: (torch.Tensor) input shape [B, N, C].

        Returns:
            x (torch.Tensor) output shape [B, N, C].
        """
        B, N, _ = x.shape

        # Normalization
        x = self.norm(x)

        qkv = self.qkv(x)
        # (B, N, num_heads, d)
        q, k, v = qkv.view(B, N, self.num_heads, -1).split(
            [self.key_dim, self.key_dim, self.attn_dim], dim=3
        )
        # (B, num_heads, N, d)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale + (
            self.attention_biases[:, self.attention_bias_idxs]
            if self.training
            else self.attn_biases
        )
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.attn_dim_all_heads)
        x = self.proj(x)
        return x
