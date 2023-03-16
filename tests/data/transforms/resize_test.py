"""Resize transformation tests."""
import torch

from vis4d.data.transforms.base import compose
from vis4d.data.transforms.resize import (
    GenerateResizeParameters,
    ResizeBoxes2D,
    ResizeImage,
)
from vis4d.data.typing import DictData


def test_resize() -> None:
    """Test resize transformation."""
    data: DictData = dict(
        cam=dict(img=torch.zeros((1, 3, 32, 32)), boxes2d=torch.ones((1, 4)))
    )
    tr1 = GenerateResizeParameters(
        shape=(16, 16), in_keys=["img"], sensors=["cam"]
    )
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
    assert tuple(data["img"].shape) == (1, 3, 16, 16)
    assert tuple(data["boxes2d"][0]) == (0.5, 0.5, 0.5, 0.5)

    data: DictData = dict(
        cam=dict(img=torch.zeros((1, 3, 32, 32)), boxes2d=torch.ones((1, 4)))
    )
    tr = compose([tr1, tr2, tr3])
    data = tr(data)["cam"]
    assert tuple(data["img"].shape) == (1, 3, 16, 16)
    assert tuple(data["boxes2d"][0]) == (0.5, 0.5, 0.5, 0.5)
