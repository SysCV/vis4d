"""Vis4D utils for distributed setting."""
from __future__ import annotations

import logging
import os
import pickle
from collections.abc import Callable
from functools import wraps
from typing import Any  # type: ignore

import cloudpickle
import torch
import torch.distributed as dist


class PicklableWrapper:
    """Wrap an object to make it more picklable.

    Note that it uses heavy weight serialization libraries that are slower than
    pickle. It's best to use it only on closures (which are usually not
    picklable). This is a simplified version of
    https://github.com/joblib/joblib/blob/master/joblib/externals/loky/cloudpickle_wrapper.py
    """

    def __init__(self, obj):
        while isinstance(obj, PicklableWrapper):
            # Wrapping an object twice is no-op
            obj = obj._obj
        self._obj = obj

    def __reduce__(self):
        s = cloudpickle.dumps(self._obj)
        return cloudpickle.loads, (s,)

    def __call__(self, *args, **kwargs):
        return self._obj(*args, **kwargs)

    def __getattr__(self, attr):
        # Ensure that the wrapped object can be used seamlessly as the previous
        # object.
        if attr not in ["_obj"]:
            return getattr(self._obj, attr)
        return getattr(self, attr)


# no coverage for these functions, since we don't unittest distributed setting
def get_world_size() -> int:  # pragma: no cover
    """Get world size of torch.distributed."""
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return int(dist.get_world_size())


def get_rank() -> int:  # pragma: no cover
    """Get global rank of torch.distributed."""
    rank_keys = ("RANK", "SLURM_PROCID", "LOCAL_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


def synchronize() -> None:  # pragma: no cover
    """Sync (barrier) among all processes when using distributed training."""
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def serialize_to_tensor(data: Any) -> torch.Tensor:  # type: ignore # pylint: disable=line-too-long # pragma: no cover
    """Serialize arbitrary picklable data to torch.Tensor."""
    backend = dist.get_backend()
    assert backend in {
        "gloo",
        "nccl",
    }, "_serialize_to_tensor only supports gloo and nccl backends."
    device = torch.device("cpu" if backend == "gloo" else "cuda")

    buffer = pickle.dumps(data)
    if len(buffer) > 1024**3:
        logger = logging.getLogger(__name__)
        logger.warning(
            "Rank %s tries all-gather %.2f GB of data on device %s",
            get_rank(),
            len(buffer) / (1024**3),
            device,
        )
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor


def rank_zero_only(func: Callable[[Any], Any]) -> Callable[[Any], Any]:  # type: ignore # pylint: disable=line-too-long
    """Function that can be used as a decorator.

    Enables a function/method being called only on global rank 0.
    """

    @wraps(func)
    def wrapped_fn(*args: Any, **kwargs: Any) -> Any:  # type: ignore
        rank = get_rank()
        if rank == 0:
            return func(*args, **kwargs)
        return None

    return wrapped_fn
