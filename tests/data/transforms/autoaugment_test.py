"""Test cases for auto-augment transform."""
import torch

from vis4d.common.util import set_random_seed
from vis4d.data.transforms.autoaugment import autoaugment


def test_autoaugment():
    """Auto augment testcase."""
    set_random_seed(0, deterministic=True)

    transform = autoaugment("original", params={})
    batch_size = 4
    x = 120 * torch.ones((batch_size, 3, 32, 32), dtype=torch.uint8)
    data_dict = {"images": x}

    x = transform(data_dict)

    assert x["images"].shape == torch.Size([batch_size, 3, 32, 32])
    assert x["images"].min() == 120
    assert x["images"].max() == 135


def test_randaugment():
    """Random augment testcase."""
    set_random_seed(0, deterministic=True)

    transform = autoaugment("rand-m9-mstd12", params={})
    batch_size = 4
    x = 120 * torch.ones((batch_size, 3, 32, 32), dtype=torch.uint8)
    data_dict = {"images": x}

    x = transform(data_dict)

    assert x["images"].shape == torch.Size([batch_size, 3, 32, 32])
    assert x["images"].min() == 120
    assert x["images"].max() == 128


def test_mixaugment():
    """Random mix augment testcase."""
    set_random_seed(0, deterministic=True)

    transform = autoaugment("augmix-m9-mstd12", params={})
    batch_size = 4
    x = 120 * torch.ones((batch_size, 3, 32, 32), dtype=torch.uint8)
    data_dict = {"images": x}

    x = transform(data_dict)

    assert x["images"].shape == torch.Size([batch_size, 3, 32, 32])
    assert x["images"].min() == 96
    assert x["images"].max() == 122
