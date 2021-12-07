"""Vis4D data samplers."""
from typing import Generator, Iterator, List, Optional, Tuple

import numpy as np
import torch
from pydantic import BaseModel
from pytorch_lightning.utilities.distributed import rank_zero_warn
from torch.utils.data import (
    BatchSampler,
    ConcatDataset,
    RandomSampler,
    Sampler,
    SequentialSampler,
)
from torch.utils.data.distributed import DistributedSampler

from vis4d.common.registry import RegistryHolder
from vis4d.common.utils import get_world_size

from .dataset import ScalabelDataset


class BaseSamplerConfig(BaseModel):
    """Base sampler config."""

    type: str
    shuffle: bool = True
    drop_last: bool = False


class BaseSampler(Sampler[List[int]], metaclass=RegistryHolder):  # type: ignore # pylint: disable=line-too-long
    """Base sampler class."""

    def __init__(
        self,
        datasets: List[ScalabelDataset],
        batch_size: int,
        drop_last: bool,
        shuffle: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        """Initialize sampler."""
        super().__init__(ConcatDataset(datasets))
        self.datasets = datasets
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.generator = generator
        self.samplers = [
            RandomSampler(dataset, generator=generator)
            if shuffle
            else SequentialSampler(dataset)
            for dataset in datasets
        ]

    def __iter__(self) -> Iterator[List[int]]:
        """Iteration method."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Return length of sampler instance."""
        raise NotImplementedError


class BaseDistributedSampler(
    DistributedSampler[List[int]], metaclass=RegistryHolder  # type: ignore
):  # pragma: no cover
    """Base distributed sampler class."""

    def __init__(
        self,
        datasets: List[ScalabelDataset],
        batch_size: int,
        drop_last: bool,
        shuffle: bool = True,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        """Initialize sampler."""
        super().__init__(
            ConcatDataset(datasets),
            num_replicas,
            rank,
            shuffle,
            seed,
            drop_last,
        )
        self.datasets = datasets
        self.batch_size = batch_size
        self.samplers = [
            DistributedSampler(
                dataset, num_replicas, rank, shuffle, seed, drop_last
            )
            for dataset in datasets
        ]

    def __iter__(self) -> Iterator[List[int]]:
        """Iteration method."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Return length of sampler instance."""
        raise NotImplementedError

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for all samplers."""
        self.epoch = epoch
        for sampler in self.samplers:
            sampler.epoch = epoch


class RoundRobin:
    """Round-robin batch-level sampling functionality."""

    @staticmethod
    def setup(
        datasets: List[ScalabelDataset],
        samplers: List[Sampler[List[int]]],
        batch_size: int,
        drop_last: bool,
    ) -> Tuple[List[Sampler[List[int]]], int, List[int]]:
        """Setup."""
        if batch_size > 1:
            samplers = [
                BatchSampler(sampler, batch_size, drop_last)
                for sampler in samplers
            ]
        max_len = max([len(sampler) for sampler in samplers])
        data_lens = [len(dataset) for dataset in datasets]
        return samplers, max_len, data_lens

    @staticmethod
    def generate_indices(
        samplers: List[Sampler[List[int]]],
        batch_size: int,
        max_len: int,
        data_lens: List[int],
    ) -> Iterator[List[int]]:
        """Generate dataset indices for each step."""
        samp_iters = [iter(sampler) for sampler in samplers]
        for _ in range(max_len):
            for i, samp_it in enumerate(samp_iters):
                batch = next(samp_it, None)
                if not batch:
                    samp_iters[i] = iter(samplers[i])
                    batch = next(samp_iters[i], None)
                assert batch is not None
                if batch_size == 1:  # pragma: no cover
                    batch = [batch]
                yield [b + sum(data_lens[:i]) for b in batch]


class RoundRobinSampler(BaseSampler):
    """Round-robin batch-level sampling (single-GPU)."""

    def __init__(
        self,
        datasets: List[ScalabelDataset],
        batch_size: int,
        drop_last: bool,
        shuffle: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        """Init."""
        super().__init__(datasets, batch_size, drop_last, shuffle, generator)
        self.samplers, self.max_len, self.data_lens = RoundRobin.setup(
            datasets, self.samplers, batch_size, drop_last
        )

    def __iter__(self) -> Iterator[List[int]]:
        """Iteration method."""
        yield from RoundRobin.generate_indices(
            self.samplers, self.batch_size, self.max_len, self.data_lens
        )

    def __len__(self) -> int:
        """Return length of sampler instance."""
        return self.max_len * len(self.samplers)


class RoundRobinDistributedSampler(BaseDistributedSampler):  # pragma: no cover
    """Round-robin batch-level sampling (distributed)."""

    def __init__(
        self,
        datasets: List[ScalabelDataset],
        batch_size: int,
        drop_last: bool,
        shuffle: bool = True,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        """Init."""
        super().__init__(datasets, batch_size, drop_last, shuffle)
        self.samplers, self.max_len, self.data_lens = RoundRobin.setup(
            datasets, self.samplers, batch_size, drop_last
        )

    def __iter__(self) -> Iterator[List[int]]:
        """Iteration method."""
        yield from RoundRobin.generate_indices(
            self.samplers, self.batch_size, self.max_len, self.data_lens
        )

    def __len__(self) -> int:
        """Return length of sampler instance."""
        return self.max_len * len(self.samplers)


def build_data_sampler(
    cfg: BaseSamplerConfig,
    datasets: List[ScalabelDataset],
    batch_size: int,
    generator: Optional[torch.Generator] = None,
) -> Sampler[List[int]]:
    """Build a sampler."""
    if get_world_size() > 1:  # pragma: no cover
        # create distributed sampler if it exists
        registry = RegistryHolder.get_registry(BaseDistributedSampler)
        registry["BaseDistributedSampler"] = BaseDistributedSampler
        dist_type = cfg.type.replace("Sampler", "DistributedSampler")
        if dist_type in registry:
            module = registry[dist_type](
                datasets, batch_size, cfg.drop_last, cfg.shuffle
            )
            assert isinstance(module, BaseDistributedSampler)
            return module
        rank_zero_warn(
            f"Distributed version of sampler {cfg.type} does not exist, "
            "adding a distributed sampler by default."
        )
    registry = RegistryHolder.get_registry(BaseSampler)
    registry["BaseSampler"] = BaseSampler
    if cfg.type in registry:
        module = registry[cfg.type](
            datasets, batch_size, cfg.drop_last, cfg.shuffle, generator
        )
        assert isinstance(module, BaseSampler)
        return module
    raise NotImplementedError(f"Sampler {cfg.type} not known!")


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
        self.sequences = list(dataset.ref_sampler.video_to_indices.keys())
        self.num_seqs = len(self.sequences)
        assert self.num_seqs >= self.num_replicas, (
            f"Number of sequences ({self.num_seqs}) must be greater or "
            f"equal to number of replicas ({self.num_replicas})!"
        )
        chunks = np.array_split(self.sequences, self.num_replicas)  # type: ignore # pylint: disable=line-too-long
        self._local_seqs = chunks[self.rank]
        self._local_idcs = []
        for seq in self._local_seqs:
            self._local_idcs.extend(dataset.ref_sampler.video_to_indices[seq])

    def __iter__(self) -> Generator[int, None, None]:
        """Iteration method."""
        yield from self._local_idcs

    def __len__(self) -> int:
        """Return length of sampler instance."""
        return len(self._local_idcs)
