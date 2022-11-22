"""Vis4D utils for distributed setting using pytorch lightening."""


from __future__ import annotations

import os
import pickle
import shutil
import tempfile
from typing import Any

import pytorch_lightning as pl
import torch

from vis4d.common.distributed import (
    get_rank,
    get_world_size,
    serialize_to_tensor,
    synchronize,
)


def _pad_to_largest_tensor(
    tensor: torch.Tensor, pl_module: pl.LightningModule
) -> tuple[list[int], torch.Tensor]:  # pragma: no cover
    """Pad tensor to largest size among the tensors in each process.

    Args:
        tensor: tensor to be padded.
        pl_module: LightningModule that contains the gathering op for the
        backend currently in use.

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
    size_list = pl_module.all_gather(local_size)
    size_list = [int(size.item()) for size in size_list]
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
    data: Any, pl_module: pl.LightningModule, rank_zero_only: bool = True
) -> list[Any] | None:  # pragma: no cover
    """Run pl_module.all_gather on arbitrary picklable data.

    Args:
        data: any picklable object
        pl_module: LightningModule that contains the gathering op for the
        backend currently in use.
        rank_zero_only: if results should only be returned on rank 0

    Returns:
        list[Any]: list of data gathered from each process
    """
    rank, world_size = get_rank(), get_world_size()
    if world_size == 1:
        return [data]

    # encode
    tensor = serialize_to_tensor(data)
    size_list, tensor = _pad_to_largest_tensor(tensor, pl_module)

    tensors = pl_module.all_gather(tensor)  # (world_size, N)

    if rank_zero_only and not rank == 0:
        return None

    # decode
    data_list = []
    for size, tensor in zip(size_list, tensors):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def create_tmpdir(
    pl_module: pl.LightningModule, rank: int, tmpdir: None | str = None
) -> str:  # pragma: no cover
    """Create and distribute a temporary directory across all processes."""
    if tmpdir is not None:
        os.makedirs(tmpdir, exist_ok=True)
        return tmpdir
    if rank == 0:
        os.makedirs(".dist_tmp", exist_ok=True)
        tmpdir = tempfile.mkdtemp(dir=".dist_tmp")
    else:
        tmpdir = None
    tmp_list = all_gather_object_gpu(tmpdir, pl_module, rank_zero_only=False)
    tmpdir = [tmp for tmp in tmp_list if tmp is not None][0]  # type: ignore
    assert isinstance(tmpdir, str), "Gather failed, tmpdir not a string!"
    return tmpdir


def all_gather_object_cpu(  # type: ignore
    data: Any,
    pl_module: pl.LightningModule,
    tmpdir: None | str = None,
    rank_zero_only: bool = True,
) -> list[Any] | None:  # pragma: no cover
    """Share arbitrary picklable data via file system caching.

    Args:
        data: any picklable object.
        pl_module: LightningModule that contains the gathering op for the
        backend currently in use.
        tmpdir: Save path for temporary files. If None, safely create tmpdir.
        rank_zero_only: if results should only be returned on rank 0

    Returns:
        list[Any]: list of data gathered from each process.
    """
    rank, world_size = get_rank(), get_world_size()
    if world_size == 1:
        return [data]

    # mk dir
    tmpdir = create_tmpdir(pl_module, rank, tmpdir)

    # encode & save
    with open(os.path.join(tmpdir, f"part_{rank}.pkl"), "wb") as f:
        pickle.dump(data, f)
    synchronize()

    if rank_zero_only and not rank == 0:
        return None

    # load & decode
    data_list = []
    for i in range(world_size):
        with open(os.path.join(tmpdir, f"part_{i}.pkl"), "rb") as f:
            data_list.append(pickle.load(f))

    # rm dir
    if not rank_zero_only:
        # wait for all processes to finish loading before removing tmpdir
        synchronize()
    if rank == 0:
        shutil.rmtree(tmpdir)

    return data_list
