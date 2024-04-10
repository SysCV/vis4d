"""Test ToTensor transform."""

import numpy as np
import torch

from vis4d.data.const import CommonKeys as K
from vis4d.data.transforms.to_tensor import ToTensor


def test_to_tensor():
    """Test the ToTensor transform."""
    data = {
        "cam": {
            K.images: np.zeros((1, 16, 16, 3)),
            "some_key": np.zeros((10, 4)),
            "my_list": [np.zeros((5, 2))],
        }
    }

    transform = ToTensor()
    data = transform.apply_to_data([data])[0][  # pylint: disable=no-member
        "cam"
    ]

    images = data[K.images]
    assert isinstance(images, torch.Tensor)
    assert images.shape == (1, 3, 16, 16)
    assert isinstance(data["some_key"], torch.Tensor)
    assert isinstance(data["my_list"][0], torch.Tensor)
