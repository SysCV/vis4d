"""Vis4D data samplers."""
from typing import Generator, Iterator, List, Optional

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


class BaseSamplerConfig(BaseModel, extra="allow"):
    """Base sampler config."""

    type: str
    shuffle: bool = True
    drop_last: bool = False


class BaseSampler(Sampler[List[int]], metaclass=RegistryHolder):  # type: ignore # pylint: disable=line-too-long
    """Base sampler class."""

    def __init__(
        self,
        dataset: ConcatDataset,
        cfg: BaseSamplerConfig,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        """Initialize sampler."""
        super().__init__(dataset)
        self.dataset = dataset
        self.cfg = cfg
        self.batch_size = batch_size
        self.drop_last = self.cfg.drop_last
        self.shuffle = self.cfg.shuffle
        self.generator = generator
        self.samplers = [
            RandomSampler(dset, generator=generator)
            if self.shuffle
            else SequentialSampler(dset)
            for dset in dataset.datasets
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
        dataset: ConcatDataset,
        cfg: BaseSamplerConfig,
        batch_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        """Initialize sampler."""
        super().__init__(
            dataset, num_replicas, rank, cfg.shuffle, seed, cfg.drop_last
        )
        self.cfg = cfg
        self.batch_size = batch_size
        self.samplers = [
            DistributedSampler(
                dset, num_replicas, rank, self.shuffle, seed, self.drop_last
            )
            for dset in dataset.datasets
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


class RoundRobinSamplerConfig(BaseSamplerConfig):
    """Round-robin sampler config."""

    repeat_sampling: bool = False  # repeat sample from exhausted data loaders
    spread_samples: bool = True  # spread samples from shorter data loaders


class RoundRobin:
    """Round-robin batch-level sampling functionality."""

    @staticmethod
    def setup(
        samplers: List[Sampler[List[int]]], batch_size: int, drop_last: bool
    ) -> List[Sampler[List[int]]]:
        """Setup."""
        if batch_size > 1:
            samplers = [
                BatchSampler(sampler, batch_size, drop_last)
                for sampler in samplers
            ]
        return samplers

    @staticmethod
    def generate_indices(
        samplers: List[Sampler[List[int]]],
        cum_sizes: List[int],
        cfg: RoundRobinSamplerConfig,
    ) -> Iterator[List[int]]:
        """Generate dataset indices for each step."""
        repeat_sampling, spread_samples = (
            cfg.repeat_sampling,
            cfg.spread_samples,
        )
        samp_iters = [iter(sampler) for sampler in samplers]
        max_len = max(len(sampler) for sampler in samplers)
        if not repeat_sampling and spread_samples:
            samp_interval = [max_len // len(sampler) for sampler in samplers]
        else:  # pragma: no cover
            if spread_samples:
                rank_zero_warn(
                    "both spread_samples and repeat_sampling are set to True"
                    ", but repeat_sampling overrides spread_samples behavior"
                )
            samp_interval = [1 for sampler in samplers]
        for e in range(max_len):
            for i, samp_it in enumerate(samp_iters):
                if e % samp_interval[i] != 0:
                    continue
                batch = next(samp_it, None)
                if batch is None:  # pragma: no cover
                    if not repeat_sampling:
                        continue
                    samp_iters[i] = iter(samplers[i])
                    batch = next(samp_iters[i], None)
                assert batch is not None
                if not isinstance(batch, list):  # pragma: no cover
                    batch = [batch]
                start_index = cum_sizes[i - 1] if i > 0 else 0
                yield [b + start_index for b in batch]

    @staticmethod
    def get_length(
        samplers: List[Sampler[List[int]]], repeat_sampling: bool = True
    ) -> int:
        """Get length of sampler."""
        sampler_lens = [len(sampler) for sampler in samplers]
        if repeat_sampling:  # pragma: no cover
            return max(sampler_lens) * len(samplers)
        return sum(sampler_lens)


class RoundRobinSampler(BaseSampler):
    """Round-robin batch-level sampling (single-GPU)."""

    def __init__(
        self,
        dataset: ConcatDataset,
        cfg: BaseSamplerConfig,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        """Init."""
        super().__init__(dataset, cfg, batch_size, generator)
        self.cfg: RoundRobinSamplerConfig = RoundRobinSamplerConfig(
            **cfg.dict()
        )
        self.samplers = RoundRobin.setup(
            self.samplers, batch_size, self.drop_last
        )

    def __iter__(self) -> Iterator[List[int]]:
        """Iteration method."""
        yield from RoundRobin.generate_indices(
            self.samplers, self.dataset.cumulative_sizes, self.cfg
        )

    def __len__(self) -> int:
        """Return length of sampler instance."""
        return RoundRobin.get_length(self.samplers, self.cfg.repeat_sampling)


class RoundRobinDistributedSampler(BaseDistributedSampler):  # pragma: no cover
    """Round-robin batch-level sampling (distributed)."""

    def __init__(
        self,
        dataset: ConcatDataset,
        cfg: BaseSamplerConfig,
        batch_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        """Init."""
        super().__init__(dataset, cfg, batch_size)
        self.cfg: RoundRobinSamplerConfig = RoundRobinSamplerConfig(
            **cfg.dict()
        )
        self.samplers = RoundRobin.setup(
            self.samplers, batch_size, self.drop_last
        )

    def __iter__(self) -> Iterator[List[int]]:
        """Iteration method."""
        yield from RoundRobin.generate_indices(
            self.samplers, self.dataset.cumulative_sizes, self.cfg
        )

    def __len__(self) -> int:
        """Return length of sampler instance."""
        return RoundRobin.get_length(self.samplers, self.cfg.repeat_sampling)


def build_data_sampler(
    cfg: BaseSamplerConfig,
    dataset: ConcatDataset,
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
            module = registry[dist_type](dataset, cfg, batch_size)
            assert isinstance(module, BaseDistributedSampler)
            return module
        rank_zero_warn(
            f"Distributed version of sampler {cfg.type} does not exist, "
            "adding a distributed sampler by default."
        )
    registry = RegistryHolder.get_registry(BaseSampler)
    registry["BaseSampler"] = BaseSampler
    if cfg.type in registry:
        module = registry[cfg.type](dataset, cfg, batch_size, generator)
        assert isinstance(module, BaseSampler)
        return module
    raise NotImplementedError(f"Sampler {cfg.type} not known!")


# no coverage for this class, since we don't unittest distributed setting
class TrackingInferenceSampler(DistributedSampler):  # type: ignore # pragma: no cover # pylint: disable=line-too-long
    """Produce sequence ordered indices for inference across all workers.

    Inference needs to run on the __exact__ set of sequences and their
    respective samples, therefore if the sequences are not divisible by the
    number of workers or if they have different length, the sampler
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
