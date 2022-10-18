"""Reference View Sampling functions.
TODO (tobias) describe what the functions are about

"""
from typing import Callable, List

import numpy as np

ViewSamplingFunc = Callable[[int, List[int]], List[int]]
SortingFunc = Callable[[int, List[int], List[int]], List[int]]


def sort_key_first(
    key_dataset_index: int, ref_indices: List[int], indices_in_video: List[int]
) -> List[int]:
    """Sort views temporally."""
    return [key_dataset_index, *ref_indices]


def sort_temporal(
    key_dataset_index: int, ref_indices: List[int], indices_in_video: List[int]
) -> List[int]:
    """Sort views temporally."""
    sorted_indices = sorted(
        [key_dataset_index, *ref_indices],
        key=lambda x: indices_in_video.index(x),
    )
    return sorted_indices


def sample_sequential(
    num_ref_samples: int, sort_views: SortingFunc = sort_key_first
) -> ViewSamplingFunc:
    """Sample reference indices sequentially."""

    def _sample(
        key_dataset_index: int, indices_in_video: List[int]
    ) -> List[int]:
        key_index = indices_in_video.index(key_dataset_index)
        right = key_index + 1 + num_ref_samples
        if right <= len(indices_in_video):
            ref_dataset_indices = indices_in_video[key_index + 1 : right]
        else:
            left = key_index - (right - len(indices_in_video))
            ref_dataset_indices = (
                indices_in_video[left:key_index]
                + indices_in_video[key_index + 1 :]
            )
        return sort_views(
            key_dataset_index, ref_dataset_indices, indices_in_video
        )

    return _sample


def sample_uniform(
    num_ref_samples: int,
    scope: int = 3,
    sort_views: SortingFunc = sort_key_first,
) -> ViewSamplingFunc:
    """Sample reference indices uniformly from neighborhood."""

    def _sample(
        key_dataset_index: int, indices_in_video: List[int]
    ) -> List[int]:
        key_index = indices_in_video.index(key_dataset_index)
        left = max(0, key_index - scope)
        right = min(key_index + scope, len(indices_in_video) - 1)
        valid_inds = (
            indices_in_video[left:key_index]
            + indices_in_video[key_index + 1 : right + 1]
        )
        ref_dataset_indices: List[int] = np.random.choice(
            valid_inds, num_ref_samples, replace=False
        ).tolist()
        return sort_views(
            key_dataset_index, ref_dataset_indices, indices_in_video
        )

    return _sample
