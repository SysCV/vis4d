"""Anchor utils."""

from __future__ import annotations

from torch import Tensor


def meshgrid(
    x_grid: Tensor, y_grid: Tensor, row_major: bool = True
) -> tuple[Tensor, Tensor]:
    """Generate mesh grid of x and y.

    Args:
        x_grid (Tensor): Grids of x dimension.
        y_grid (Tensor): Grids of y dimension.
        row_major (bool, optional): Whether to return y grids first.
            Defaults to True.

    Returns:
        tuple[Tensor]: The mesh grids of x and y.
    """
    # use shape instead of len to keep tracing while exporting to onnx
    xx = x_grid.repeat(y_grid.shape[0])
    yy = y_grid.view(-1, 1).repeat(1, x_grid.shape[0]).view(-1)
    if row_major:
        return xx, yy
    return yy, xx
