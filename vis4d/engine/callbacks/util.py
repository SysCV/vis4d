"""Utility functions for callbacks."""
from __future__ import annotations

from vis4d.config.util import ConfigDict, instantiate_classes

from .base import Callback
from .visualizer import VisualizerCallback


def instantiate_callbacks(
    callbacks_cfg: list[ConfigDict], visualize: bool = False
) -> list[Callback]:
    """Instantiate callbacks.

    Args:
        callbacks_cfg (list[ConfigDict]): List of callback configurations.
        visualize (bool, optional): Whether to visualize predictions.
            Defaults to False.
    """
    callbacks = []
    for cb in callbacks_cfg:
        cb = instantiate_classes(cb)

        if isinstance(cb, VisualizerCallback) and not visualize:
            continue

        callbacks.append(cb)

    return callbacks
