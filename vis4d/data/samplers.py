"""Vis4D data samplers."""
from typing import Generator, Iterator, List, Optional

import numpy as np
import torch
from torch.utils.data import BatchSampler, RandomSampler, Sampler
from torch.utils.data.distributed import DistributedSampler

from vis4d.common.registry import RegistryHolder

from .dataset import ScalabelDataset


class BaseSampler(Sampler[List[int]], metaclass=RegistryHolder):  # type: ignore # pylint: disable=line-too-long
    """Base sampler class."""

    def __init__(
        self,
        datasets: List[ScalabelDataset],
        batch_size: int,
        drop_last: bool,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        """Initialize sampler."""
        super().__init__(None)
        self.datasets = datasets
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.generator = generator
        self.samplers = [
            RandomSampler(dataset, generator=generator) for dataset in datasets
        ]

    def __iter__(self) -> Iterator[List[int]]:
        """Iteration method."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Return length of sampler instance."""
        raise NotImplementedError


class RoundRobinSampler(BaseSampler):
    """Round-robin batch-level sampling."""

    def __init__(
        self,
        datasets: List[ScalabelDataset],
        batch_size: int,
        drop_last: bool,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        """Init."""
        super().__init__(datasets, batch_size, drop_last, generator)
        if batch_size > 1:
            self.samplers = [
                BatchSampler(sampler, batch_size, drop_last)
                for sampler in self.samplers
            ]
        self.max_len = max([len(sampler) for sampler in self.samplers])
        self.data_lens = [len(dataset) for dataset in self.datasets]

    def __iter__(self) -> Iterator[List[int]]:
        """Iteration method."""
        samp_iters = [iter(sampler) for sampler in self.samplers]
        for _ in range(self.max_len):
            for i, samp_it in enumerate(samp_iters):
                batch = next(samp_it, None)
                if not batch:
                    samp_iters[i] = iter(self.samplers[i])
                    batch = next(samp_iters[i], None)
                assert batch is not None
                if self.batch_size == 1:
                    batch = [batch]
                yield [b + sum(self.data_lens[:i]) for b in batch]

    def __len__(self) -> int:
        """Return length of sampler instance."""
        return self.max_len * len(self.samplers)


def build_data_sampler(
    sampler_name: str,
    datasets: List[ScalabelDataset],
    batch_size: int,
    drop_last: bool = False,
    generator: Optional[torch.Generator] = None,
) -> BaseSampler:
    """Build a sampler."""
    registry = RegistryHolder.get_registry(BaseSampler)
    if sampler_name in registry:
        module = registry[sampler_name](
            datasets, batch_size, drop_last, generator
        )
        assert isinstance(module, BaseSampler)
        return module
    raise NotImplementedError(f"Sampler {sampler_name} not known!")


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
        chunks = np.array_split(self.sequences, self.num_replicas)  # type: ignore # pylint: disable=line-too-long
        self._local_seqs = chunks[self.rank]
        self._local_idcs = []
        for seq in self._local_seqs:
            self._local_idcs.extend(dataset.video_to_indices[seq])

    def __iter__(self) -> Generator[int, None, None]:
        """Iteration method."""
        yield from self._local_idcs

    def __len__(self) -> int:
        """Return length of sampler instance."""
        return len(self._local_idcs)
