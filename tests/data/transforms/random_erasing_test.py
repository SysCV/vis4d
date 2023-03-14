"""Test cases for normalize transform."""
import torch

from vis4d.data.transforms.random_erasing import random_erasing


def test_random_erasing() -> None:
    """Random erasing testcase."""
    transform = random_erasing(
        probability=1.0,
        min_area=0.25,
        max_area=0.25,
        min_aspect_ratio=1,
        max_aspect_ratio=1,
        mean=(1.0, 0.0, 0.0),
    )
    batch_size = 4
    x = torch.zeros((batch_size, 3, 10, 10), dtype=torch.float32)
    data_dict = {"images": x}

    x = transform(data_dict)

    # The sum of the image should be 25, which is the number of pixels erased,
    # regardless of the location of the erased region.
    for i in range(batch_size):
        assert torch.isclose(x["images"][i].sum(), torch.tensor(25.0))
        assert torch.isclose(x["images"][i, 1:].sum(), torch.tensor(0.0))

    assert x["images"].shape == torch.Size([batch_size, 3, 10, 10])
