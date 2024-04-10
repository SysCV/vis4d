"""Vis4D data samplers."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

from .datasets.base import VideoDataset
from .typing import DictDataOrList


class VideoInferenceSampler(
    DistributedSampler[list[int]]
):  # pragma: no cover # No unittest for distributed setting.
    """Produce sequence ordered indices for inference across all workers.

    Inference needs to run on the __exact__ set of sequences and their
    respective samples, therefore if the sequences are not divisible by the
    number of workers or if they have different length, the sampler
    produces different number of samples on different workers.
    """

    def __init__(
        self,
        dataset: Dataset[DictDataOrList],
        num_replicas: None | int = None,
        rank: None | int = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        """Creates an instance of the class.

        Args:
            dataset (Dataset): Inference dataset.
            num_replicas (int, optional): Number of processes participating in
                distributed training. By default, :attr:`world_size` is
                retrieved from the current distributed group.
            rank (int, optional): Rank of the current process within
                :attr:`num_replicas`. By default, :attr:`rank` is retrieved
                from the current distributed group.
            shuffle (bool, optional): If ``True`` (default), sampler will
                shuffle the indices.
            seed (int, optional): random seed used to shuffle the sampler if
                :attr:`shuffle=True`. This number should be identical across
                all processes in the distributed group. Default: ``0``.
            drop_last (bool, optional): if ``True``, then the sampler will drop
                the tail of the data to make it evenly divisible across the
                number of replicas. If ``False``, the sampler will add extra
                indices to make the data evenly divisible across the replicas.
                Default: ``False``.
        """
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        assert isinstance(dataset, VideoDataset)
        self.sequences = list(dataset.video_mapping["video_to_indices"])
        self.num_seqs = len(self.sequences)
        assert self.num_seqs >= self.num_replicas, (
            f"Number of sequences ({self.num_seqs}) must be greater or "
            f"equal to number of replicas ({self.num_replicas})!"
        )
        chunks = np.array_split(self.sequences, self.num_replicas)
        self._local_seqs = chunks[self.rank]
        self._local_idcs: list[int] = []
        for seq in self._local_seqs:
            self._local_idcs.extend(
                dataset.video_mapping["video_to_indices"][seq]
            )

    def __iter__(self) -> Iterator[list[int]]:
        """Iteration method."""
        return iter(self._local_idcs)  # type: ignore

    def __len__(self) -> int:
        """Return length of sampler instance."""
        return len(self._local_idcs)
