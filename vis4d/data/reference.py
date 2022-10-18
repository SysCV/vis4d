"""Reference View Sampling.

These Classes sample reference views from a dataset that contains videos.
This is usually used when a model needs multiple samples of a video during
training.
"""
from abc import abstractmethod
from typing import Callable, List

import numpy as np

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


class ReferenceViewSampler:
    """Base reference view sampler."""

    def __init__(
        self, num_ref_samples: int, sort_fn: SortingFunc = sort_key_first
    ) -> None:
        """Init.

        Args:
            num_ref_samples (int): Number of reference views to sample.
            sort_fn (SortingFunc, optional): Function that sorts the views. Defaults to sort_key_first.
        """
        self.sort_fn = sort_fn
        self.num_ref_samples = num_ref_samples

    @abstractmethod
    def _sample_ref_indices(
        self, key_index: int, indices_in_video: List[int]
    ) -> List[int]:
        """Sample num_ref_samples reference view indices.

        Args:
            key_index (int): Index of key view in the video
            indices_in_video (List[int]): all dataset indices in the video

        Returns:
            List[int]: dataset indices of reference views.
        """
        raise NotImplementedError

    def __call__(
        self, key_dataset_index: int, indices_in_video: List[int]
    ) -> List[int]:
        """Call function. Wraps _sample_ref_indices with sorting."""
        key_index = indices_in_video.index(key_dataset_index)
        ref_indices = self._sample_ref_indices(key_index, indices_in_video)
        return self.sort_fn(key_dataset_index, ref_indices, indices_in_video)


class SequentialViewSampler(ReferenceViewSampler):
    """Sequential View Sampler."""

    def _sample_ref_indices(
        self, key_index: int, indices_in_video: List[int]
    ) -> List[int]:
        """Sample sequential reference views."""
        right = key_index + 1 + self.num_ref_samples
        if right <= len(indices_in_video):
            ref_dataset_indices = indices_in_video[key_index + 1 : right]
        else:
            left = key_index - (right - len(indices_in_video))
            ref_dataset_indices = (
                indices_in_video[left:key_index]
                + indices_in_video[key_index + 1 :]
            )
        return ref_dataset_indices


class UniformViewSampler(ReferenceViewSampler):
    """View Sampler that chooses reference views uniform at random."""

    def __init__(
        self,
        scope: int,
        num_ref_samples: int,
        sort_fn: SortingFunc = sort_key_first,
    ) -> None:
        """Init.

        Args:
            scope (int): Define scope of neighborhood to key view to sample from.
            num_ref_samples (int): Number of reference views to sample.
            sort_fn (SortingFunc, optional): Function that sorts the views. Defaults to sort_key_first.
        """
        super().__init__(num_ref_samples, sort_fn)
        self.scope = scope

    def _sample_ref_indices(
        self, key_index: int, indices_in_video: List[int]
    ) -> List[int]:
        """Uniformly sample reference views."""
        left = max(0, key_index - self.scope)
        right = min(key_index + self.scope, len(indices_in_video) - 1)
        valid_inds = (
            indices_in_video[left:key_index]
            + indices_in_video[key_index + 1 : right + 1]
        )
        ref_dataset_indices: List[int] = np.random.choice(
            valid_inds, self.num_ref_samples, replace=False
        ).tolist()
        return ref_dataset_indices
