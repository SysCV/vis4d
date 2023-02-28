# mypy: disable-error-code=misc
"""This module contains utilities for multiprocess parallelism."""
from __future__ import annotations

import logging
import os
import pickle
import shutil
import tempfile
from functools import wraps
from typing import Any

import cloudpickle
import torch
import torch.distributed as dist
from typing_extensions import Protocol


class AnyCallback(Protocol):
    """Protocol for callback with any arguments."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore
        """Call."""


class PicklableWrapper:  #  mypy: disable=line-too-long
    """Wrap an object to make it more picklable.

    Note that it uses heavy weight serialization libraries that are slower than
    pickle. It's best to use it only on closures (which are usually not
    picklable). This is a simplified version of
    https://github.com/joblib/joblib/blob/master/joblib/externals/loky/cloudpickle_wrapper.py
    """

    def __init__(self, obj: Any | PicklableWrapper) -> None:  # type: ignore
        """Creates an instance of the class."""
        while isinstance(obj, PicklableWrapper):
            # Wrapping an object twice is no-op
            obj = obj._obj
        self._obj: Any = obj

    def __reduce__(self) -> tuple[Any, tuple[bytes]]:
        """Reduce."""
        s = cloudpickle.dumps(self._obj)
        return cloudpickle.loads, (s,)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call."""
        return self._obj(*args, **kwargs)

    def __getattr__(self, attr: str) -> Any:
        """Get attribute.

        Ensure that the wrapped object can be used seamlessly as the previous
        object.
        """
        if attr not in ["_obj"]:
            return getattr(self._obj, attr)
        return getattr(self, attr)


# no coverage for these functions, since we don't unittest distributed setting
def get_world_size() -> int:  # pragma: no cover
    """Get the world size (number of processes) of torch.distributed.

    Returns:
        int: The world size.
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return int(dist.get_world_size())


def get_rank() -> int:  # pragma: no cover
    """Get the global rank of the current process in torch.distributed.

    Returns:
        int: The global rank.
    """
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID")
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


def serialize_to_tensor(data: Any) -> torch.Tensor:  # pragma: no cover
    """Serialize arbitrary picklable data to a torch.Tensor.

    Args:
        data (Any): The data to serialize.

    Returns:
        torch.Tensor: The serialized data as a torch.Tensor.

    Raises:
        AssertionError: If the backend of torch.distributed is not gloo or
            nccl.
    """
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


def rank_zero_only(func: AnyCallback) -> AnyCallback:
    """Allows the decorated function to be called only on global rank 0.

    Args:
        func( Callable[[Any], Any]): The function to decorate.

    Returns:
        Callable[[Any], Any]: The decorated function.

    """

    @wraps(func)
    def wrapped_fn(*args: Any, **kwargs: Any) -> Any:
        rank = get_rank()
        if rank == 0:
            return func(*args, **kwargs)
        return None

    return wrapped_fn


def pad_to_largest_tensor(
    tensor: torch.Tensor,
) -> tuple[list[int], torch.Tensor]:  # pragma: no cover
    """Pad tensor to largest size among the tensors in each process.

    Args:
        tensor: tensor to be padded.

    Returns:
        list[int]: size of the tensor, on each rank
        Tensor: padded tensor that has the max size
    """
    world_size = get_world_size()
    assert (
        world_size >= 1
    ), "_pad_to_largest_tensor requires distributed setting!"
    local_size = torch.tensor(
        [tensor.numel()], dtype=torch.int64, device=tensor.device
    )
    local_size_list = [local_size.clone() for _ in range(world_size)]
    dist.all_gather_object(local_size_list, local_size)
    size_list = [int(size.item()) for size in local_size_list]
    max_size = max(size_list)

    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    if local_size != max_size:
        padding = torch.zeros(
            (max_size - local_size,), dtype=torch.uint8, device=tensor.device
        )
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor


def all_gather_object_gpu(  # type: ignore
    data: Any, rank_zero_return_only: bool = True
) -> list[Any] | None:  # pragma: no cover
    """Run pl_module.all_gather on arbitrary picklable data.

    Args:
        data: any picklable object
        rank_zero_return_only: if results should only be returned on rank 0

    Returns:
        list[Any]: list of data gathered from each process
    """
    rank, world_size = get_rank(), get_world_size()
    if world_size == 1:
        return [data]

    # encode
    tensor = serialize_to_tensor(data)
    size_list, tensor = pad_to_largest_tensor(tensor)
    tensor_list = [tensor.clone() for _ in range(world_size)]
    dist.all_gather_object(tensor_list, tensor)  # (world_size, N)

    if rank_zero_return_only and not rank == 0:
        return None

    # decode
    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def create_tmpdir(
    rank: int, tmpdir: None | str = None
) -> str:  # pragma: no cover
    """Create and distribute a temporary directory across all processes."""
    if tmpdir is not None:
        os.makedirs(tmpdir, exist_ok=True)
        return tmpdir
    if rank == 0:
        # create a temporary directory
        default_tmpdir = tempfile.gettempdir()
        if default_tmpdir is not None:
            dir_path = os.path.join(default_tmpdir, ".dist_tmp")
        else:
            dir_path = ".dist_tmp"
        os.makedirs(dir_path, exist_ok=True)
        tmpdir = tempfile.mkdtemp(dir=dir_path)
    else:
        tmpdir = None
    tmp_list = all_gather_object_gpu(tmpdir, rank_zero_return_only=False)
    tmpdir = [tmp for tmp in tmp_list if tmp is not None][0]  # type: ignore
    assert isinstance(tmpdir, str), "Gather failed, tmpdir not a string!"
    return tmpdir


def all_gather_object_cpu(  # type: ignore
    data: Any,
    tmpdir: None | str = None,
    rank_zero_return_only: bool = True,
) -> list[Any] | None:  # pragma: no cover
    """Share arbitrary picklable data via file system caching.

    Args:
        data: any picklable object.
        tmpdir: Save path for temporary files. If None, safely create tmpdir.
        rank_zero_return_only: if results should only be returned on rank 0

    Returns:
        list[Any]: list of data gathered from each process.
    """
    rank, world_size = get_rank(), get_world_size()
    if world_size == 1:
        return [data]

    # mk dir
    tmpdir = create_tmpdir(rank, tmpdir)

    # encode & save
    with open(os.path.join(tmpdir, f"part_{rank}.pkl"), "wb") as f:
        pickle.dump(data, f)
    synchronize()

    if rank_zero_return_only and not rank == 0:
        return None

    # load & decode
    data_list = []
    for i in range(world_size):
        with open(os.path.join(tmpdir, f"part_{i}.pkl"), "rb") as f:
            data_list.append(pickle.load(f))

    # rm dir
    if not rank_zero_return_only:
        # wait for all processes to finish loading before removing tmpdir
        synchronize()
    if rank == 0:
        shutil.rmtree(tmpdir)

    return data_list
