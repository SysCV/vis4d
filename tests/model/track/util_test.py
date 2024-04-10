"""Utility functions for track module test."""

from vis4d.model.track.util import split_key_ref_indices


def test_split_key_ref_indices():
    """Test split key and reference indices."""
    # Key 1st , batch size of 1
    keyframes = [[True], [False]]

    key_ind, ref_inds = split_key_ref_indices(keyframes)

    assert key_ind == 0
    assert ref_inds == [1]

    # Key 1st, batch size of 2
    keyframes = [[True, True], [False, False]]

    key_ind, ref_inds = split_key_ref_indices(keyframes)

    assert key_ind == 0
    assert ref_inds == [1]

    # Key 3rd, batch size of 2
    keyframes = [[False, False], [False, False], [True, True]]

    key_ind, ref_inds = split_key_ref_indices(keyframes)

    assert key_ind == 2
    assert ref_inds == [0, 1]
