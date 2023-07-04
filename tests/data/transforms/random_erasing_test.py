"""Test cases for normalize transform."""
import numpy as np

from vis4d.data.transforms.random_erasing import RandomErasing


def test_random_erasing() -> None:
    """Random erasing testcase."""
    transform = RandomErasing(
        min_area=0.25,
        max_area=0.25,
        min_aspect_ratio=1,
        max_aspect_ratio=1,
        mean=(1.0, 0.0, 0.0),
    )
    batch_size = 4
    x = np.zeros((batch_size, 10, 10, 3), dtype=np.float32)

    x = transform([x])[0]

    # The sum of the image should be 25, which is the number of pixels erased,
    # regardless of the location of the erased region.
    for i in range(batch_size):
        assert np.isclose(x[i].sum().item(), 25.0)
        assert np.isclose(x[i, :, :, 1:].sum().item(), 0.0)

    assert x.shape == (batch_size, 10, 10, 3)


def test_random_erasing_bypass() -> None:
    """Random erasing testcase."""
    transform = RandomErasing(
        min_area=0,
        max_area=0,
    )
    batch_size = 4
    x_ori = np.random.randn(batch_size, 10, 10, 3).astype(np.float32)

    x = transform([x_ori])[0]

    assert (x == x_ori).all()
