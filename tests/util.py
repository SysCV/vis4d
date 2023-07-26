"""Utilities for unit tests."""
from __future__ import annotations

import inspect
import os

import numpy as np
import torch
from torch import nn

from vis4d.common.typing import ModelOutput, NDArrayI64, NDArrayNumber
from vis4d.engine.connectors import LossConnector, data_key
from vis4d.engine.loss_module import LossModule

MOCKLOSS = LossModule(
    {
        "loss": nn.L1Loss(),
        "connector": LossConnector({"test": data_key("test")}),
    }
)


def fill_weights(module: nn.Module, value: float = 0.0) -> None:
    """Fill weights of an nn.Module with specific value.

    Enables deterministic exeuction of computation for testing purposes.

    Args:
        module (nn.Module): The module.
        value (float, optional): The desired value. Defaults to 0.0.
    """
    for param in module.parameters():
        param.data.fill_(value)


def get_test_data(dir_name: str) -> str:
    """Return the absolute path to the given test data directory.

    Args:
        dir_name (str): Name of test data directory.

    Returns:
        str: Absolute path to test data directory.
    """
    prefix = os.path.dirname(os.path.abspath(inspect.stack()[1][1]))
    prefix_code, _ = prefix.rsplit("tests", 1)
    return os.path.join(prefix_code, "tests", "vis4d-test-data", dir_name)


def get_test_file(file_name: str, rel_path: None | str = None) -> str:
    """Return the absolute path to the given test file.

    The test file is assumed to be in a 'testcases' folder in tests, possibly
    with identical relative path to the current file.
    Args:
        file_name (str): Name of the test file.
        rel_path (str, optional): Relative path inside test directory.
            Defaults to None.

    Returns:
        str: Absolute path to test file <cwd/testcases/file_name>.
    """
    prefix = os.path.dirname(os.path.abspath(inspect.stack()[1][1]))
    prefix_code, prefix_rel = prefix.rsplit("tests", 1)
    if rel_path is None:
        rel_path = prefix_rel
    return os.path.join(
        prefix_code, "tests", rel_path.strip("/"), "testcases", file_name
    )


def generate_features(
    channels: int,
    init_height: int,
    init_width: int,
    num_features: int,
    batch_size: int = 1,
    double_channels: bool = False,
) -> list[torch.Tensor]:
    """Create a random list of features maps with decreasing size.

    Args:
        channels (int): Number of feature channels (C).
        init_height (int): Target feature map height (h).
        init_width (int): Target feature map width (w).
        num_features (int): Number of features maps.
        batch_size (int, optional): Batch size (B)
        double_channels (bool, optional): If channels should be doubled for
                                          each feature map.

    Returns:
        list[torch.Tensor]: List containing feature tensors
                            shaped [B, C', h/(2^i), w/(2^i)], where i is
                            the position in de list and C' is either C or
                            C*(2^i) depending if double_channels is true
    """
    state = torch.random.get_rng_state()
    torch.random.set_rng_state(torch.manual_seed(0).get_state())

    features_list = []
    channel_factor = 1
    for i in range(num_features):
        features_list.append(
            torch.rand(
                batch_size,
                channels * channel_factor,
                init_height // (2**i),
                init_width // (2**i),
            )
        )
        if double_channels:
            channel_factor *= 2

    torch.random.set_rng_state(state)
    return features_list


def generate_features_determ(
    channels: int,
    init_height: int,
    init_width: int,
    num_features: int,
    batch_size: int = 1,
    double_channels: bool = False,
) -> list[torch.Tensor]:
    """Create a deterministic list of features maps with decreasing size.

    Uses torch.arange instead of torch.rand so the final features will be
    fixed no matter the randomness.

    Args:
        channels (int): Number of feature channels (C).
        init_height (int): Target feature map height (h).
        init_width (int): Target feature map width (w).
        num_features (int): Number of features maps.
        batch_size (int, optional): Batch size (B)
        double_channels (bool, optional): If channels should be doubled for
                                          each feature map.

    Returns:
        list[torch.Tensor]: List containing feature tensors
                            shaped [B, C', h/(2^i), w/(2^i)], where i is
                            the position in de list and C' is either C or
                            C*(2^i) depending if double_channels is true
    """
    features_list = []
    channel_factor = 1
    for i in range(num_features):
        channel = channels * channel_factor
        height, width = init_height // (2**i), init_width // (2**i)
        dims = [batch_size, channel, height, width]
        features_list.append(
            torch.arange(np.prod(dims)).reshape(*dims) / (np.prod(dims))
        )
        if double_channels:
            channel_factor *= 2

    return features_list


def generate_boxes(
    height: int,
    width: int,
    num_boxes: int,
    batch_size: int = 1,
    track_ids: bool = False,
    use_score: bool = True,
) -> tuple[
    list[torch.Tensor],
    list[torch.Tensor | None],
    list[torch.Tensor],
    list[torch.Tensor | None],
]:
    """Generate random detection boxes.

    Args:
        height (int): Image height
        width (int): Image width
        num_boxes (int): Number of boxes to load
        batch_size (int, optional): Batch size
        track_ids (bool, optional): If track ids should be loaded.
        use_score (bool, optional): If scores should be loaded.

    Returns:
        tuple[list[torch.Tensor] x 4]: [bounding boxes], [scores], [num_boxes],
                                       [track_ids].
    """
    state = torch.random.get_rng_state()
    torch.random.set_rng_state(torch.manual_seed(0).get_state())
    if use_score:
        box = [width, height, width, height, 1.0]
    else:
        box = [width, height, width, height]
    rand_max = torch.repeat_interleave(torch.tensor([box]), num_boxes, dim=0)
    box_tensor = torch.rand(num_boxes, 5 if use_score else 4) * rand_max
    sorted_xy = [
        box_tensor[:, [0, 2]].sort(dim=-1)[0],
        box_tensor[:, [1, 3]].sort(dim=-1)[0],
    ]
    box_tensor[:, :4] = torch.cat(
        [
            sorted_xy[0][:, 0:1],
            sorted_xy[1][:, 0:1],
            sorted_xy[0][:, 1:2],
            sorted_xy[1][:, 1:2],
        ],
        dim=-1,
    )
    tracks = torch.arange(0, num_boxes) if track_ids else None
    torch.random.set_rng_state(state)
    return (
        [box_tensor[:, :-1]] * batch_size,
        [box_tensor[:, -1:] if use_score else None] * batch_size,
        [torch.zeros(num_boxes, dtype=torch.long)] * batch_size,
        [tracks] * batch_size,
    )


def generate_instance_masks(
    height: int,
    width: int,
    num_masks: int,
    batch_size: int = 1,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Generate random instance masks.

    Args:
        height (int): Image height
        width (int): Image width
        num_masks (int): Amount of masks to generate
        batch_size (int, optional): Batch size

    Returns:
        (list[torch.Tensor] x 3): [masks], [scores], [id]
    """
    state = torch.random.get_rng_state()
    torch.random.set_rng_state(torch.manual_seed(0).get_state())
    rand_mask = torch.randint(0, num_masks, (height, width))
    mask_tensor = torch.stack(
        [(rand_mask == i).type(torch.uint8) for i in range(num_masks)]
    )
    torch.random.set_rng_state(state)
    return (
        [mask_tensor] * batch_size,
        [torch.rand(num_masks)] * batch_size,
        [torch.arange(num_masks)] * batch_size,
    )


def generate_semantic_masks(
    height: int,
    width: int,
    num_classes: int,
    batch_size: int = 1,
) -> list[torch.Tensor]:
    """Generate random semantic masks.

    Args:
        height (int): Image height
        width (int): Image width
        num_classes (int): Number of classes
        batch_size (int, optional): Batch size

    Returns:
        list[torch.Tensor]: [masks]
    """
    state = torch.random.get_rng_state()
    torch.random.set_rng_state(torch.manual_seed(0).get_state())
    rand_mask = torch.randint(0, num_classes, (height, width))
    torch.random.set_rng_state(state)
    return [rand_mask] * batch_size


def isclose_on_all_indices_tensor(
    tensor: torch.Tensor,
    indices: torch.Tensor,
    expected: torch.Tensor,
    atol: float = 1e-4,
    rtol: float = 1e-6,
) -> bool:
    """Check if values from two tensors are close enough on indices.

    Args:
        tensor (torch.Tensor): Input tensor.
        indices (torch.Tensor): Indices of values in tensors to compare.
        expected (torch.Tensor): Expected tensor.
        atol (float, optional): Absolute tolerance. Defaults to 1e-4.
        rtol (float, optional): Relative tolerance. Defaults to 1e-6.

    Returns:
        bool: True if values are close enough, False otherwise.
    """
    if not torch.all(torch.isfinite(tensor)):
        return False
    if not torch.all(torch.isfinite(expected)):
        return False
    if tensor[indices].shape != expected.shape:
        return False
    return torch.allclose(
        tensor[indices],
        expected,
        atol=atol,
        rtol=rtol,
    )


def isclose_on_all_indices_numpy(
    tensor: NDArrayNumber,
    indices: NDArrayI64,
    expected: NDArrayNumber,
    atol: float = 1e-4,
    rtol: float = 1e-6,
) -> bool:
    """Check if values from two numpy arrays are close enough on indices.

    Args:
        tensor (np.ndarray): Input array.
        indices (np.ndarray): Indices of values in arrays to compare.
        expected (np.ndarray): Expected array.
        atol (float, optional): Absolute tolerance. Defaults to 1e-4.
        rtol (float, optional): Relative tolerance. Defaults to 1e-6.

    Returns:
        bool: True if values are close enough, False otherwise.
    """
    if not np.all(np.isfinite(tensor)):
        return False
    if not np.all(np.isfinite(expected)):
        return False
    if tensor[indices].shape != expected.shape:
        return False
    return np.allclose(
        tensor[indices],
        expected,
        atol=atol,
        rtol=rtol,
    )


class MockModel(nn.Module):
    """Model Mockup."""

    def __init__(self, model_param: int, *args, **kwargs):  # type: ignore # pylint: disable=unused-argument,line-too-long
        """Creates an instance of the class."""
        super().__init__()
        self.model_param = model_param
        self.linear = nn.Linear(10, 1)

    def forward(  # type: ignore
        self, *args, **kwargs  # pylint: disable=unused-argument,line-too-long
    ) -> ModelOutput:
        """Forward."""
        if self.training:
            return {
                "my_loss": (
                    self.linear(
                        torch.rand((1, 10), device=self.linear.weight.device)
                    )
                    - 0
                ).sum()
            }
        return {}
