"""Utility functions for model zoo tests."""
from __future__ import annotations

import importlib

from vis4d.config import FieldConfigDict


def get_config_for_name(config_name: str) -> FieldConfigDict:
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
            will be ignored.
            Defaults to [].
    """
    if ignored_props is None:
        ignored_props = []

    lines1 = content1.splitlines()
    lines2 = content2.splitlines()
    if len(lines1) != len(lines2):
        print("File length missmatch:", len(lines1), "!=", len(lines2))
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
                "Line missmatch #", line_id, ":", line1, "!=", lines2[line_id]
            )
            return False

    return True
