# pylint: disable=no-member,unexpected-keyword-arg,use-dict-literal
"""Resize transformation tests."""
import numpy as np

from vis4d.data.transforms.base import compose
from vis4d.data.transforms.resize import (
    GenerateResizeParameters,
    ResizeBoxes2D,
    ResizeImage,
    ResizeInstanceMasks,
    ResizeIntrinsics,
    ResizeSegMasks,
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


def test_resize_instance_masks():
    """Test resize instance masks."""
    masks = np.array(
        [
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1],
                [0, 0, 1, 1, 1],
                [0, 0, 1, 1, 1],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1],
                [0, 0, 1, 1, 1],
                [0, 0, 1, 1, 1],
            ],
        ]
    )
    target_shape = (3, 3)
    expected = np.array(
        [
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 1],
            ],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 1],
            ],
        ]
    )
    transform = ResizeInstanceMasks()
    result = transform(masks, target_shape)
    assert (result == expected).all()


def test_resize_seg_masks():
    """Test resize segmentation masks."""
    masks = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 2, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    target_shape = (3, 3)
    expected = np.array(
        [
            [0, 0, 0],
            [0, 1, 1],
            [0, 1, 1],
        ]
    )
    transform = ResizeSegMasks()
    result = transform(masks, target_shape)
    assert (result == expected).all()


def test_resize_intrinsics():
    """Test resize intrinsics."""
    intrinsics = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    scale_factor = (0.5, 0.5)
    expected_intrinsics = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]])
    resize_intrinsics = ResizeIntrinsics()
    assert np.allclose(
        resize_intrinsics(intrinsics, scale_factor), expected_intrinsics
    )
