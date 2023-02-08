"""Resize transformation tests."""
import torch

from vis4d.data.const import CommonKeys
from vis4d.data.transforms.resize import Resize
from vis4d.data.typing import DictData


def test_resize() -> None:
    """Test resize transformation."""
    data: DictData = dict(
        cam=dict(img=torch.zeros((1, 3, 32, 32)), boxes2d=torch.ones((1, 4)))
    )
    x = Resize((16, 16), sensors="cam")
    x.remap(CommonKeys.images, "img")
    data = x(data)["cam"]

    # both img and boxes2d are resized now
    assert tuple(data["img"].shape) == (1, 3, 16, 16)
    assert tuple(data["boxes2d"][0]) == (0.5, 0.5, 0.5, 0.5)

    # change desired shape now, see if parameter sharing works
    x.shape = (8, 8)
    data: DictData = dict(
        cam=dict(
            img=torch.zeros((1, 3, 32, 32)),
            boxes2d=torch.ones((1, 4)),
            transforms=data["transforms"],
        ),
    )
    data = x(data)["cam"]

    # it should not still resize to 16, 16
    assert tuple(data["img"].shape) == (1, 3, 16, 16)
    assert tuple(data["boxes2d"][0]) == (0.5, 0.5, 0.5, 0.5)

    # if we turn it off, now it should resize to 8, 8
    data: DictData = dict(
        cam=dict(img=torch.zeros((1, 3, 32, 32)), boxes2d=torch.ones((1, 4)))
    )
    data = x(data)["cam"]
    assert tuple(data["img"].shape) == (1, 3, 8, 8)
    assert tuple(data["boxes2d"][0]) == (0.25, 0.25, 0.25, 0.25)
