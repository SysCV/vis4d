# pylint: disable=no-member,unexpected-keyword-arg,use-dict-literal
"""Resize transformation tests."""
import numpy as np

from vis4d.data.transforms.base import compose
from vis4d.data.transforms.resize import (
    GenResizeParameters,
    ResizeBoxes2D,
    ResizeImage,
)
from vis4d.data.typing import DictData


def test_resize() -> None:
    """Test resize transformation."""
    data: DictData = dict(
        cam=dict(
            img=np.zeros((1, 32, 32, 3), dtype=np.float32),
            boxes2d=np.ones((1, 4), dtype=np.float32),
        )
    )
    tr1 = GenResizeParameters(shape=(16, 16), in_keys=["img"], sensors=["cam"])
    data = tr1.apply_to_data(data)
    tr2 = ResizeImage(
        in_keys=[
            "img",
            "transforms.resize.target_shape",
            "transforms.resize.interpolation",
        ],
        out_keys=["img"],
        sensors=["cam"],
    )
    data = tr2.apply_to_data(data)
    tr3 = ResizeBoxes2D(sensors=["cam"])
    data = tr3.apply_to_data(data)["cam"]
    assert tuple(data["img"].shape) == (1, 16, 16, 3)
    assert tuple(data["boxes2d"][0]) == (0.5, 0.5, 0.5, 0.5)

    data: DictData = dict(
        cam=dict(
            img=np.zeros((1, 32, 32, 3), dtype=np.float32),
            boxes2d=np.ones((1, 4), dtype=np.float32),
        )
    )
    tr = compose([tr1, tr2, tr3])
    data = tr(data)["cam"]
    assert tuple(data["img"].shape) == (1, 16, 16, 3)
    assert tuple(data["boxes2d"][0]) == (0.5, 0.5, 0.5, 0.5)
