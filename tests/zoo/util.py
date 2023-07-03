"""Utility functions for model zoo tests."""
from __future__ import annotations

import importlib

from vis4d.config.typing import ExperimentConfig


def get_config_for_name(config_name: str) -> ExperimentConfig:
    """Get config for name."""
    module = importlib.import_module("vis4d.zoo." + config_name)

    return module.get_config()


def content_equal(
    content1: str, content2: str, ignored_props: list[str] | None = None
) -> bool:
    """Compare two strings line by line.

    Args:
        content1 (str): First file content
        content2 (str): Second file content
        ignored_props (list[str], optional): List of properties to ignore.
            All lines matching any of these properties (followed by a ':')
            will be ignored. Defaults to [].
    """
    if ignored_props is None:
        ignored_props = []

    lines1 = content1.splitlines()
    lines2 = content2.splitlines()
    if len(lines1) != len(lines2):
        print("File length mismatch:", len(lines1), "!=", len(lines2))
        return False

    for line_id, line1 in enumerate(lines1):
        skip = False
        for prop in ignored_props:
            # Append `:` to avoid matching a property that is a substring of
            # another property
            prop = prop + ":"
            if prop in line1 and prop in lines2[line_id]:
                skip = True
                continue  # ignore these lines

        if not skip and line1 != lines2[line_id]:
            print(
                "Line mismatch #", line_id, ":", line1, "!=", lines2[line_id]
            )
            return False

    return True


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

    import os

    if not os.path.isfile(cfg_gt):
        # write config to file
        with open(cfg_gt, "w", encoding="UTF-8") as f:
            f.write(config)

    with open(cfg_gt, "r", encoding="UTF-8") as f:
        gt_config = f.read()

    return content_equal(config, gt_config, varying_keys)
