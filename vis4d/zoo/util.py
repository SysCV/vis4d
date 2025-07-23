"""Utility functions for the zoo module."""

from __future__ import annotations

import importlib

from vis4d.config.typing import ExperimentConfig


def get_config_for_name(config_name: str) -> ExperimentConfig:
    """Get config for name."""
    module = importlib.import_module("vis4d.zoo." + config_name)

    return module.get_config()
