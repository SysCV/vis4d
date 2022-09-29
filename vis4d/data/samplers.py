"""Vis4D data samplers."""
from typing import Generator, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from torch.utils.data import (
    BatchSampler,
    ConcatDataset,
    RandomSampler,
    Sampler,
    SequentialSampler,
)
from torch.utils.data.distributed import DistributedSampler

from vis4d.common_to_revise.registry import RegistryHolder
from vis4d.common_to_revise.utils import get_world_size
from vis4d.struct_to_revise import ArgsType, ModuleCfg

from .datasets import BaseVideoDataset


class BaseSampler(Sampler[List[int]], metaclass=RegistryHolder):  # type: ignore # pylint: disable=line-too-long
    """Base sampler class."""

    def __init__(
        self,
        dataset: ConcatDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        """Initialize sampler."""
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
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
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        """Initialize sampler."""
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
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


class RoundRobin:
    """Round-robin batch-level sampling functionality."""

    @staticmethod
    def setup_parameters(
        samplers: List[Sampler[List[int]]],
        repeat_interval: Union[int, List[int]],
        spread_samples: Union[bool, List[bool]],
        max_samples: Union[int, List[int]],
    ) -> Tuple[List[int], List[bool], List[int]]:
        """Setup sampler parameters."""
        if isinstance(repeat_interval, int):
            repeat_interval_ = [repeat_interval] * len(samplers)
        else:
            assert len(repeat_interval) == len(samplers)
            repeat_interval_ = repeat_interval
        if isinstance(spread_samples, bool):
            spread_samples_ = [spread_samples] * len(samplers)
        else:
            assert len(spread_samples) == len(samplers)
            spread_samples_ = spread_samples
        if isinstance(max_samples, int):
            max_samples_ = [max_samples] * len(samplers)
        else:
            assert len(max_samples) == len(samplers)
            max_samples_ = max_samples
        return repeat_interval_, spread_samples_, max_samples_

    @staticmethod
    def setup_samplers(
        samplers: List[Sampler[List[int]]], batch_size: int, drop_last: bool
    ) -> List[Sampler[List[int]]]:
        """Setup samplers."""
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
        repeat_interval: List[int],
        spread_samples: List[bool],
        max_samples: List[int],
    ) -> Iterator[List[int]]:
        """Generate dataset indices for each step."""
        samp_iters = [iter(sampler) for sampler in samplers]
        samp_lens = RoundRobin.get_sampler_lens(samplers, max_samples)
        samp_interval = RoundRobin.get_samp_intervals(
            samplers, samp_lens, repeat_interval, spread_samples
        )
        for e in range(max(samp_lens)):
            for i, samp_it in enumerate(samp_iters):
                if samp_interval[i] != 0 and e % samp_interval[i] != 0:
                    continue
                batch = next(samp_it, None)
                if batch is None:
                    if repeat_interval[i] == 0:
                        continue
                    samp_iters[i] = iter(samplers[i])
                    batch = next(samp_iters[i], None)
                assert batch is not None
                if not isinstance(batch, list):
                    batch = [batch]
                start_index = cum_sizes[i - 1] if i > 0 else 0
                yield [b + start_index for b in batch]

    @staticmethod
    def get_sampler_lens(
        samplers: List[Sampler[List[int]]], max_samples: List[int]
    ) -> List[int]:
        """Get length of each sampler."""
        return [
            len(sampler)
            if max_samples[i] == -1
            else min(len(sampler), max_samples[i])
            for i, sampler in enumerate(samplers)
        ]

    @staticmethod
    def get_samp_intervals(
        samplers: List[Sampler[List[int]]],
        samp_lens: List[int],
        repeat_interval: List[int],
        spread_samples: List[bool],
    ) -> List[int]:
        """Get length of each sampler."""
        samp_interval, max_len = [], max(samp_lens)
        for i in range(len(samplers)):
            if repeat_interval[i] == 0 and spread_samples[i]:
                samp_interval.append(max_len // samp_lens[i])
                continue
            if spread_samples[i]:
                rank_zero_warn(
                    "both spread_samples and repeat_interval are set to True,"
                    " but repeat_interval overrides spread_samples behavior"
                )
            samp_interval.append(repeat_interval[i])
        return samp_interval

    @staticmethod
    def get_length(
        samplers: List[Sampler[List[int]]],
        repeat_interval: List[int],
        max_samples: List[int],
    ) -> int:
        """Get length of round-robin sampler."""
        sampler_lens = RoundRobin.get_sampler_lens(samplers, max_samples)
        total_len = 0
        for i, _ in enumerate(samplers):
            if repeat_interval[i] > 0:
                total_len += max(sampler_lens) // repeat_interval[i]
            else:
                total_len += sampler_lens[i]
        return total_len


class RoundRobinSampler(BaseSampler):
    """Round-robin batch-level sampling (single-GPU)."""

    def __init__(
        self,
        *args: ArgsType,
        repeat_interval: Union[int, List[int]] = 0,
        spread_samples: Union[bool, List[bool]] = True,
        max_samples: Union[int, List[int]] = -1,
        **kwargs: ArgsType,
    ) -> None:
        """Init."""
        super().__init__(*args, **kwargs)
        (
            self.repeat_interval,
            self.spread_samples,
            self.max_samples,
        ) = RoundRobin.setup_parameters(
            self.samplers, repeat_interval, spread_samples, max_samples
        )
        self.samplers = RoundRobin.setup_samplers(
            self.samplers, self.batch_size, self.drop_last
        )

    def __iter__(self) -> Iterator[List[int]]:
        """Iteration method."""
        yield from RoundRobin.generate_indices(
            self.samplers,
            self.dataset.cumulative_sizes,
            self.repeat_interval,
            self.spread_samples,
            self.max_samples,
        )

    def __len__(self) -> int:
        """Return length of sampler instance."""
        return RoundRobin.get_length(
            self.samplers, self.repeat_interval, self.max_samples
        )


class RoundRobinDistributedSampler(BaseDistributedSampler):  # pragma: no cover
    """Round-robin batch-level sampling (distributed)."""

    def __init__(
        self,
        *args: ArgsType,
        repeat_interval: Union[int, List[int]] = 0,
        spread_samples: Union[bool, List[bool]] = True,
        max_samples: Union[int, List[int]] = -1,
        **kwargs: ArgsType,
    ) -> None:
        """Init."""
        super().__init__(*args, **kwargs)
        (
            self.repeat_interval,
            self.spread_samples,
            self.max_samples,
        ) = RoundRobin.setup_parameters(
            self.samplers, repeat_interval, spread_samples, max_samples
        )
        self.samplers = RoundRobin.setup_samplers(
            self.samplers, self.batch_size, self.drop_last
        )

    def __iter__(self) -> Iterator[List[int]]:
        """Iteration method."""
        yield from RoundRobin.generate_indices(
            self.samplers,
            self.dataset.cumulative_sizes,
            self.repeat_interval,
            self.spread_samples,
            self.max_samples,
        )

    def __len__(self) -> int:
        """Return length of sampler instance."""
        return RoundRobin.get_length(
            self.samplers, self.repeat_interval, self.max_samples
        )


def build_data_sampler(
    cfg: ModuleCfg,
    dataset: ConcatDataset,
    batch_size: int,
    generator: Optional[torch.Generator] = None,
) -> Sampler[List[int]]:
    """Build a sampler."""
    sampler_type = cfg.pop("type", None)
    if sampler_type is None:
        raise ValueError(f"Need type argument in sampler config: {cfg}")
    if get_world_size() > 1:  # pragma: no cover
        # create distributed sampler if it exists
        registry = RegistryHolder.get_registry(BaseDistributedSampler)
        registry["BaseDistributedSampler"] = BaseDistributedSampler
        dist_type = sampler_type.replace("Sampler", "DistributedSampler")
        if dist_type in registry:
            module = registry[dist_type](dataset, batch_size, **cfg)
            assert isinstance(module, BaseDistributedSampler)
            return module
        rank_zero_warn(
            f"Distributed version of sampler {dist_type} does not exist, "
            "adding a distributed sampler by default."
        )
    registry = RegistryHolder.get_registry(BaseSampler)
    registry["BaseSampler"] = BaseSampler
    if sampler_type in registry:
        module = registry[sampler_type](
            dataset, batch_size, **cfg, generator=generator
        )
        assert isinstance(module, BaseSampler)
        return module
    raise NotImplementedError(f"Sampler {sampler_type} not known!")


# no coverage for this class, since we don't unittest distributed setting
class VideoInferenceSampler(DistributedSampler):  # type: ignore # pragma: no cover # pylint: disable=line-too-long
    """Produce sequence ordered indices for inference across all workers.

    Inference needs to run on the __exact__ set of sequences and their
    respective samples, therefore if the sequences are not divisible by the
    number of workers or if they have different length, the sampler
    produces different number of samples on different workers.
    """

    def __init__(
        self,
        dataset: BaseVideoDataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        """Init."""
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.sequences = list(dataset.video_to_indices)
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
