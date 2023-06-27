# pylint: disable=no-member,unexpected-keyword-arg
"""Test base transforms."""
from __future__ import annotations

import copy

import numpy as np

from vis4d.data.const import CommonKeys as K
from vis4d.data.transforms.base import RandomApply, compose
from vis4d.data.transforms.flip import FlipImages
from vis4d.data.transforms.resize import (
    GenerateResizeParameters,
    ResizeBoxes2D,
    ResizeImages,
)


def test_random_apply():
    """Test random apply."""
    data = {K.images: np.random.rand(1, 16, 16, 3)}

    random_apply = RandomApply([FlipImages()], 1.0)

    data_tr = random_apply.apply_to_data([copy.deepcopy(data)])[0]

    assert (data_tr[K.images] == data[K.images][:, :, ::-1, :]).all()

    random_apply = RandomApply([FlipImages()], probability=0.0)

    data_tr = random_apply.apply_to_data([copy.deepcopy(data)])[0]

    assert (data_tr[K.images] == data[K.images]).all()


def test_compose():
    """Test compose."""
    data = {
        "cam": {
            "img": np.zeros((1, 32, 32, 3), dtype=np.float32),
            "boxes2d": np.ones((1, 4), dtype=np.float32),
        }
    }

    tr1 = GenerateResizeParameters(
        shape=(16, 16), in_keys=["img"], sensors=["cam"]
    )
    tr2 = ResizeImages(
        in_keys=[
            "img",
            "transforms.resize.target_shape",
            "transforms.resize.interpolation",
        ],
        out_keys=["img"],
        sensors=["cam"],
    )
    tr3 = ResizeBoxes2D(sensors=["cam"])

    tr = compose([tr1, tr2, tr3])
    data = tr([copy.deepcopy(data)])[0]["cam"]
    assert tuple(data["img"].shape) == (1, 16, 16, 3)
    assert tuple(data["boxes2d"][0]) == (0.5, 0.5, 0.5, 0.5)
