"""Test cases for auto-augment transform."""
import torch

from vis4d.common.util import set_random_seed
from vis4d.data.transforms.autoaugment import augmix, autoaug, randaug


def test_autoaugment() -> None:
    """Auto augment testcase."""
    set_random_seed(0, deterministic=True)

    transform = autoaug(policy="original", magnitude_std=0.5)
    batch_size = 4
    x = 120 * torch.ones((batch_size, 3, 32, 32), dtype=torch.uint8)
    data_dict = {"images": x}

    x = transform(data_dict)

    assert x["images"].shape == torch.Size([batch_size, 3, 32, 32])
    assert x["images"].min() == 120
    assert x["images"].max() == 128


def test_randaugment() -> None:
    """Random augment testcase."""
    set_random_seed(0, deterministic=True)

    transform = randaug(magnitude=9, magnitude_std=0.5)
    batch_size = 4
    x = 120 * torch.ones((batch_size, 3, 32, 32), dtype=torch.uint8)
    data_dict = {"images": x}

    x = transform(data_dict)

    assert x["images"].shape == torch.Size([batch_size, 3, 32, 32])
    assert x["images"].min() == 120
    assert x["images"].max() == 128


def test_randaugment_increasing() -> None:
    """Random augment with increasing transforms testcase."""
    set_random_seed(0, deterministic=True)

    transform = randaug(magnitude=9, use_increasing=True, magnitude_std=0.5)
    batch_size = 4
    x = 120 * torch.ones((batch_size, 3, 32, 32), dtype=torch.uint8)
    data_dict = {"images": x}

    x = transform(data_dict)

    assert x["images"].shape == torch.Size([batch_size, 3, 32, 32])
    assert x["images"].min() == 120
    assert x["images"].max() == 128


def test_mixaugment() -> None:
    """Random mix augment testcase."""
    set_random_seed(0, deterministic=True)

    transform = augmix(magnitude=9, magnitude_std=12, alpha=0.5)
    batch_size = 4
    x = 120 * torch.ones((batch_size, 3, 32, 32), dtype=torch.uint8)
    data_dict = {"images": x}

    x = transform(data_dict)

    assert x["images"].shape == torch.Size([batch_size, 3, 32, 32])
    assert x["images"].min() == 48
    assert x["images"].max() == 125
