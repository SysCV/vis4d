# pylint: disable=no-member,unexpected-keyword-arg
"""Test cases for normalize transform."""
import torch

import numpy as np

from vis4d.data.transforms.normalize import (
    NormalizeImage,
    BatchNormalizeImages,
)


def test_normalize():
    """Image normalize testcase."""
    data = {
        "cam": {
            "images": np.zeros((1, 12, 12, 3), dtype=np.float32),
            "boxes2d": np.ones((1, 4), dtype=np.float32),
        }
    }
    tr1 = NormalizeImage(sensors=["cam"])
    data = tr1.apply_to_data(data)
    img = torch.from_numpy(data["cam"]["images"]).permute(0, 3, 1, 2)

    assert torch.isclose(
        img.view(3, -1).mean(dim=-1),
        torch.tensor([-2.1179, -2.0357, -1.8044]),
        rtol=0.0001,
    ).all()

    batch_data = [
        {
            "cam": {
                "img": np.zeros((1, 12, 12, 3), dtype=np.float32),
                "boxes2d": np.ones((1, 4), dtype=np.float32),
            }
        }
    ]
    tr2 = BatchNormalizeImages(
        in_keys=["img"],
        out_keys=["img"],
        sensors=["cam"],
    )
    batch_data = tr2.apply_to_data(batch_data)

    img = torch.from_numpy(batch_data[0]["cam"]["img"]).permute(0, 3, 1, 2)

    assert torch.isclose(
        img.view(3, -1).mean(dim=-1),
        torch.tensor([-2.1179, -2.0357, -1.8044]),
        rtol=0.0001,
    ).all()
