"""VisT data Samplers."""
from typing import Generator, Optional

from torch.utils.data.distributed import DistributedSampler

from .dataset import ScalabelDataset


# no coverage for this class, since we don't unittest distributed setting
class TrackingInferenceSampler(DistributedSampler):  # type: ignore # pragma: no cover # pylint: disable=line-too-long
    """Produce sequence ordered indices for inference across all workers.

    Inference needs to run on the __exact__ set of sequences and their
    respective samples, therefore if the sequences are not divisible by the
    number of workers of if they have different length, the sampler
    produces different number of samples on different workers.
    """

    def __init__(
        self,
        dataset: ScalabelDataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        """Init."""
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.sequences = list(dataset.video_to_indices.keys())
        self.num_seqs = len(self.sequences)
        assert self.num_seqs >= self.num_replicas, (
            f"Number of sequences ({self.num_seqs}) must be greater or "
            f"equal to number of replicas ({self.num_replicas})!"
        )

        shard_size = (self.num_seqs - 1) // self.num_replicas + 1
        begin = shard_size * self.rank
        end = min(shard_size * (self.rank + 1), self.num_seqs)
        self._local_idcs = []
        for i in range(begin, end):
            self._local_idcs.extend(
                dataset.video_to_indices[self.sequences[i]]
            )

    def __iter__(self) -> Generator[int, None, None]:
        """Iteration method."""
        yield from self._local_idcs

    def __len__(self) -> int:
        """Return length of sampler instance."""
        return len(self._local_idcs)
