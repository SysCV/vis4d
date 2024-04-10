"""Utility functions for track module."""

from __future__ import annotations


def split_key_ref_indices(
    keyframes: list[list[bool]],
) -> tuple[int, list[int]]:
    """Get key frame from list of sample attributes."""
    key_ind = None
    ref_inds = []
    for i, is_keys in enumerate(keyframes):
        assert all(
            is_keys[0] == is_key for is_key in is_keys
        ), "Same batch should have the same view."
        if is_keys[0]:
            key_ind = i
        else:
            ref_inds.append(i)

    assert key_ind is not None, "Key frame not found."
    assert len(ref_inds) > 0, "No reference frames found."

    return key_ind, ref_inds
