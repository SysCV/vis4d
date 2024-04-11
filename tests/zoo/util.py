"""Utility functions for model zoo tests."""

from __future__ import annotations

import importlib

from tests.util import content_equal
from vis4d.zoo.typing import ExperimentConfig


def get_config_for_name(config_name: str) -> ExperimentConfig:
    """Get config for name."""
    module = importlib.import_module("vis4d.zoo." + config_name)

    return module.get_config()


def compare_configs(
    cfg_cur: str, cfg_gt: str, varying_keys: list[str] | None = None
) -> bool:
    """Compare two configs.

    Args:
        cfg_cur (str): Path to current config.
        cfg_gt (str): Path to ground truth config.
        varying_keys (list[str], optional): List of keys that are allowed to
            vary. Defaults to None.
    """
    config = get_config_for_name(cfg_cur).to_yaml()

    with open(cfg_gt, "r", encoding="UTF-8") as f:
        gt_config = f.read()

    return content_equal(config, gt_config, varying_keys)
