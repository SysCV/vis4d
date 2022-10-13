"""Testcases for reference view sampling."""
from .reference import sort_temporal


def test_temporal_sorting():
    """Test temporal sorting function."""
    # dataset level indices
    key, refs = 51, [64, 94, 10]

    # video indices
    video = [29, 40, 10, 3, 64, 94, 78, 51]

    sorted_indices = sort_temporal(key, refs, video)
    assert tuple(sorted_indices) == (10, 64, 94, 51)
