# pylint: disable=no-member,unexpected-keyword-arg
"""Test cases for normalize transform."""
import numpy as np
import torch

from vis4d.data.transforms.normalize import NormalizeImages


def test_normalize():
    """Image normalize testcase."""
    data = {
        "cam": {
            "images": np.zeros((1, 12, 12, 3), dtype=np.float32),
            "boxes2d": np.ones((1, 4), dtype=np.float32),
        }
    }
    transform = NormalizeImages(sensors=["cam"])
    data = transform.apply_to_data([data])[0]
    img = torch.from_numpy(data["cam"]["images"]).permute(0, 3, 1, 2)

    assert torch.isclose(
        img.view(3, -1).mean(dim=-1),
        torch.tensor([-2.1179, -2.0357, -1.8044]),
        rtol=0.0001,
    ).all()
