"""Reference View Sampling.

These Classes sample reference views from a dataset that contains videos.
This is usually used when a model needs multiple samples of a video during
training.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Callable, List

import numpy as np
from torch.utils.data import Dataset

from .const import CommonKeys as K
from .datasets import VideoDataset
from .typing import DictData

SortingFunc = Callable[[int, List[int], List[int]], List[int]]


def sort_key_first(
    key_dataset_index: int,
    ref_indices: list[int],
    indices_in_video: list[int],  # pylint: disable=unused-argument
) -> list[int]:
    """Sort views temporally."""
    return [key_dataset_index, *ref_indices]


def sort_temporal(
    key_dataset_index: int, ref_indices: list[int], indices_in_video: list[int]
) -> list[int]:
    """Sort views temporally."""
    sorted_indices = sorted(
        [key_dataset_index, *ref_indices],
        key=indices_in_video.index,
    )
    return sorted_indices


class ReferenceViewSampler:
    """Base reference view sampler."""

    def __init__(self, num_ref_samples: int) -> None:
        """Creates an instance of the class.

        Args:
            num_ref_samples (int): Number of reference views to sample.
        """
        self.num_ref_samples = num_ref_samples

    @abstractmethod
    def _sample_ref_indices(
        self, key_index: int, indices_in_video: list[int]
    ) -> list[int]:
        """Sample num_ref_samples reference view indices.

        Args:
            key_index (int): Index of key view in the video
            indices_in_video (list[int]): all dataset indices in the video

        Returns:
            list[int]: dataset indices of reference views.
        """
        raise NotImplementedError

    def __call__(
        self, key_dataset_index: int, indices_in_video: list[int]
    ) -> list[int]:
        """Call function. Wraps _sample_ref_indices with sorting."""
        key_index = indices_in_video.index(key_dataset_index)
        return self._sample_ref_indices(key_index, indices_in_video)


class SequentialViewSampler(ReferenceViewSampler):
    """Sequential View Sampler."""

    def _sample_ref_indices(
        self, key_index: int, indices_in_video: list[int]
    ) -> list[int]:
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

    def __init__(self, scope: int, num_ref_samples: int) -> None:
        """Creates an instance of the class.

        Args:
            scope (int): Define scope of neighborhood to key view to sample
                from.
            num_ref_samples (int): Number of reference views to sample.
        """
        super().__init__(num_ref_samples)
        if scope != 0 and scope < num_ref_samples // 2:
            raise ValueError("Scope must be higher than num_ref_imgs / 2.")
        self.scope = scope

    def _sample_ref_indices(
        self, key_index: int, indices_in_video: list[int]
    ) -> list[int]:
        """Uniformly sample reference views."""
        if self.scope > 0:
            left = max(0, key_index - self.scope)
            right = min(key_index + self.scope, len(indices_in_video) - 1)
            valid_inds = (
                indices_in_video[left:key_index]
                + indices_in_video[key_index + 1 : right + 1]
            )
            ref_dataset_indices: list[int] = np.random.choice(
                valid_inds, self.num_ref_samples, replace=False
            ).tolist()
        else:
            ref_dataset_indices = [key_index]
        return ref_dataset_indices


class MultiViewDataset(Dataset[list[DictData]]):
    """Dataset that samples reference views from a video dataset."""

    def __init__(
        self,
        dataset: VideoDataset,
        sampler: ReferenceViewSampler,
        skip_nomatch_samples: bool = False,
    ) -> None:
        """Creates an instance of the class.

        Args:
            dataset (Dataset): Video dataset to sample from.
            sampler (ReferenceViewSampler): Sampler that samples reference
                views.
            skip_nomatch_samples (bool, optional): Whether to skip samples
                where no match is found. Defaults to False.
        """
        self.dataset = dataset
        self.sampler = sampler
        self.skip_nomatch_samples = skip_nomatch_samples

    @staticmethod
    def has_matches(
        key_data: DictData,
        ref_data: list[DictData],
        match_key: str = K.boxes2d_track_ids,
    ) -> bool:
        """Check if key / ref data have matches."""
        key_target = key_data[match_key]
        for ref_view in ref_data:
            ref_target = ref_view[match_key]
            match = np.equal(
                np.expand_dims(key_target, axis=1), ref_target[None]
            )
            if match.any():
                return True
        return False  # pragma: no cover

    def __len__(self) -> int:
        """Get length of dataset."""
        return len(self.dataset)

    def get_video_indices(self, index: int) -> list[int]:
        """Get indices of videos in dataset."""
        for indices in self.dataset.video_to_indices.values():
            if index in indices:
                return indices
        raise ValueError(f"Index {index} not found in video_to_indices!")

    # TODO: Implement sorting. Currently always key first.
    def __getitem__(self, index: int) -> list[DictData]:
        """Get item from dataset."""
        cur_sample = self.dataset[index]
        cur_sample["keyframes"] = True

        if self.sampler.num_ref_samples > 0:
            ref_data = []

            if (
                isinstance(self.sampler, UniformViewSampler)
                and self.sampler.scope == 0
            ):
                ref_indices = [index] * self.sampler.num_ref_samples
            else:
                video_indices = self.get_video_indices(index)
                ref_indices = self.sampler(index, video_indices)

            for ref_index in ref_indices:
                ref_sample = self.dataset[ref_index]
                ref_sample["keyframes"] = False
                ref_data.append(ref_sample)

            if self.skip_nomatch_samples and not (
                self.has_matches(cur_sample, ref_data)
            ):
                # TODO: implement retry
                raise NotImplementedError

            assert self.sampler.num_ref_samples == len(ref_data)
            return [cur_sample, *ref_data]
        return [cur_sample]
