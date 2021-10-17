"""Utilities for augmentation."""
from typing import Tuple, Union

import torch
from torch.distributions import Bernoulli

TensorWithTransformMat = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


def identity_matrix(inputs: torch.Tensor) -> torch.Tensor:
    """Return 3x3 identity matrix."""
    return torch.eye(3, device=inputs.device, dtype=inputs.dtype)


def adapted_sampling(
    shape: Union[Tuple, torch.Size],  # type: ignore
    dist: torch.distributions.Distribution,
    same_on_batch: bool = False,
) -> torch.Tensor:
    """The uniform sampling function that accepts 'same_on_batch'.

    If same_on_batch is True, all values generated will be exactly same given
    a batch_size (shape[0]). By default, same_on_batch is set to False.
    """
    if same_on_batch:
        return dist.sample((1, *shape[1:])).repeat(
            shape[0], *[1] * (len(shape) - 1)
        )
    return dist.sample(shape)


def batch_prob_generator(
    batch_shape: torch.Size,
    prob: float,
    prob_batch: float,
    same_on_batch: bool,
) -> torch.Tensor:
    """Generate probability of augmentation for batch."""
    batch_prob: torch.Tensor
    if prob_batch == 1:
        batch_prob = torch.tensor([True])
    elif prob_batch == 0:
        batch_prob = torch.tensor([False])
    else:
        batch_prob = adapted_sampling(
            (1,), Bernoulli(prob_batch), same_on_batch
        ).bool()

    if batch_prob.sum().item() == 1:
        elem_prob: torch.Tensor
        if prob == 1:
            elem_prob = torch.tensor([True] * batch_shape[0])
        elif prob == 0:
            elem_prob = torch.tensor([False] * batch_shape[0])
        else:
            elem_prob = adapted_sampling(
                (batch_shape[0],), Bernoulli(prob), same_on_batch
            ).bool()
        batch_prob = batch_prob * elem_prob
    else:
        batch_prob = batch_prob.repeat(batch_shape[0])
    return batch_prob
