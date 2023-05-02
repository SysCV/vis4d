"""Test cases for auto-augment transform."""
import numpy as np

from vis4d.common.util import set_random_seed
from vis4d.data.transforms.autoaugment import (
    AugMix,
    AutoAugOriginal,
    AutoAugV0,
    RandAug,
)


def test_autoaugment() -> None:
    """Auto augment testcase."""
    set_random_seed(0, deterministic=True)

    transform = AutoAugOriginal(magnitude_std=0.5)
    batch_size = 4
    x = 120 * np.ones((batch_size, 32, 32, 3), dtype=np.uint8)

    x = transform(x)

    assert x.shape == (batch_size, 32, 32, 3)
    assert x.min() == 120
    assert x.max() == 128


def test_autoaugment_v0() -> None:
    """Auto augment testcase."""
    set_random_seed(0, deterministic=True)

    transform = AutoAugV0(magnitude_std=0.5)
    batch_size = 4
    x = 128 * np.ones((batch_size, 32, 32, 3), dtype=np.uint8)

    x = transform(x)

    assert x.shape == (batch_size, 32, 32, 3)
    assert x.min() == 127
    assert x.max() == 128


def test_randaugment() -> None:
    """Random augment testcase."""
    set_random_seed(0, deterministic=True)

    transform = RandAug(magnitude=10, magnitude_std=0.5)
    batch_size = 4
    x = 120 * np.ones((batch_size, 32, 32, 3), dtype=np.uint8)

    x = transform(x)

    assert x.shape == (batch_size, 32, 32, 3)
    assert x.min() == 120
    assert x.max() == 128


def test_randaugment_increasing() -> None:
    """Random augment with increasing transforms testcase."""
    set_random_seed(0, deterministic=True)

    transform = RandAug(magnitude=9, use_increasing=True, magnitude_std=0.5)
    batch_size = 4
    x = 120 * np.ones((batch_size, 32, 32, 3), dtype=np.uint8)

    x = transform(x)

    assert x.shape == (batch_size, 32, 32, 3)
    assert x.min() == 120
    assert x.max() == 128


def test_mixaugment() -> None:
    """Random mix augment testcase."""
    set_random_seed(0, deterministic=True)

    transform = AugMix(magnitude=9, magnitude_std=12, alpha=0.5)
    batch_size = 4
    x = 120 * np.ones((batch_size, 32, 32, 3), dtype=np.uint8)

    x = transform(x)

    assert x.shape == (batch_size, 32, 32, 3)
    assert x.min() == 48
    assert x.max() == 125
