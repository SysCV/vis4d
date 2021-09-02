"""VisT utils for distributed setting."""
import logging
import pickle
from typing import Any, List, Tuple

import pytorch_lightning as pl
import torch
import torch.distributed as dist

TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])


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
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return int(dist.get_rank())


def synchronize() -> None:  # pragma: no cover
    """Sync (barrier) among all processes when using distributed training."""
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    if dist.get_backend() == dist.Backend.NCCL and TORCH_VERSION >= (1, 8):
        # This argument is needed to avoid warnings.
        # It's valid only for NCCL backend.
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


def _serialize_to_tensor(data: Any) -> torch.Tensor:  # type: ignore # pylint: disable=line-too-long # pragma: no cover
    """Serialize arbitrary picklable data to torch.Tensor."""
    backend = dist.get_backend()
    assert backend in [
        "gloo",
        "nccl",
    ], "_serialize_to_tensor only supports gloo and nccl backends."
    device = torch.device("cpu" if backend == "gloo" else "cuda")

    buffer = pickle.dumps(data)
    if len(buffer) > 1024 ** 3:
        logger = logging.getLogger(__name__)
        logger.warning(
            "Rank {} tries all-gather {:.2f} GB of data on device {}".format(
                get_rank(), len(buffer) / (1024 ** 3), device
            )
        )
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor


def _pad_to_largest_tensor(
    tensor: torch.Tensor, pl_module: pl.LightningModule
) -> Tuple[List[int], torch.Tensor]:  # pragma: no cover
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


def all_gather_object(data: Any, pl_module: pl.LightningModule) -> List[Any]:  # type: ignore # pylint: disable=line-too-long # pragma: no cover
    """Run pl_module.all_gather on arbitrary picklable data.

    Args:
        data: any picklable object
        pl_module: LightningModule that contains the gathering op for the
        backend currently in use.

    Returns:
        List[Any]: list of data gathered from each process
    """
    if get_world_size() == 1:
        return [data]

    # encode
    tensor = _serialize_to_tensor(data)
    size_list, tensor = _pad_to_largest_tensor(tensor, pl_module)

    tensors = pl_module.all_gather(tensor)  # (world_size, N)

    # decode
    data_list = []
    for size, tensor in zip(size_list, tensors):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list
