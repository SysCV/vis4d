"""VisT data Samplers."""
from typing import Generator

from torch.utils.data.sampler import Sampler

from ..common.utils import get_rank, get_world_size
from .dataset import ScalabelDataset


class TrackingInferenceSampler(Sampler):  # type: ignore
    """Produce sequence ordered indices for inference across all workers.

    Inference needs to run on the __exact__ set of sequences and their
    respective samples, therefore if the sequences are not divisible by the
    number of workers of if they have different length, the sampler
    produces different number of samples on different workers.
    """

    def __init__(self, dataset: ScalabelDataset):
        """Init."""
        super().__init__(None)
        self._sequences = list(dataset.video_to_indices.keys())
        self._num_seqs = len(self._sequences)
        self._rank = get_rank()
        self._world_size = get_world_size()
        assert self._num_seqs >= self._world_size, (
            f"Number of sequences ({self._num_seqs}) must be greater or "
            f"equal to number of replicas ({self._world_size})!"
        )

        shard_size = (self._num_seqs - 1) // self._world_size + 1
        begin = shard_size * self._rank
        end = min(shard_size * (self._rank + 1), self._num_seqs)
        self._local_idcs = []
        for i in range(begin, end):
            self._local_idcs.extend(
                dataset.video_to_indices[self._sequences[i]]
            )

    def __iter__(self) -> Generator[int, None, None]:
        """Iteration method."""
        yield from self._local_idcs

    def __len__(self) -> int:
        """Return length of sampler instance."""
        return len(self._local_idcs)
