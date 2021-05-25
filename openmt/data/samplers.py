"""OpenMT data Samplers."""
from typing import Generator

from detectron2.utils import comm
from torch.utils.data.sampler import Sampler

from .dataset_mapper import MapDataset


class TrackingInferenceSampler(Sampler):  # type: ignore
    """Produce indices for inference across all workers.

    Inference needs to run on the __exact__ set of samples, therefore when
    the total number of samples is not divisible by the number of workers,
    this sampler produces different number of samples on different workers.
    """

    def __init__(self, dataset: MapDataset):
        """Init."""
        super().__init__(None)
        self._sequences = list(dataset.video_to_indices.keys())
        self._num_seqs = len(self._sequences)
        assert self._num_seqs > 0
        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

        shard_size = (self._num_seqs - 1) // self._world_size + 1
        begin = shard_size * self._rank
        end = min(shard_size * (self._rank + 1), self._num_seqs)
        self._local_idcs = []
        for i in range(begin, end):
            self._local_idcs.extend(dataset.video_to_indices[self._sequences[i]])

    def __iter__(self) -> Generator[int, None, None]:
        """Iteration method."""
        yield from self._local_idcs

    def __len__(self) -> int:
        """Return length of sampler instance."""
        return len(self._local_idcs)
