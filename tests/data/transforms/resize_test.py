# pylint: disable=no-member,unexpected-keyword-arg,use-dict-literal
"""Resize transformation tests."""
import copy
import unittest

import numpy as np

from vis4d.data.const import CommonKeys as K
from vis4d.data.transforms.resize import (
    GenerateResizeParameters,
    ResizeBoxes2D,
    ResizeImages,
    ResizeInstanceMasks,
    ResizeIntrinsics,
    ResizeSegMasks,
)
from vis4d.data.typing import DictData


class TestResize(unittest.TestCase):
    """Test resize transformation."""

    data: DictData = dict(
        cam=dict(
            img=np.zeros((1, 32, 32, 3), dtype=np.float32),
            boxes2d=np.ones((1, 4), dtype=np.float32),
        )
    )

    def test_resize(self) -> None:
        """Test resize transformation."""
        tr1 = GenerateResizeParameters(
            shape=(16, 16), in_keys=["img"], sensors=["cam"]
        )
        data = tr1.apply_to_data([copy.deepcopy(self.data)])
        tr2 = ResizeImages(
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
        data = tr3.apply_to_data(data)[0]["cam"]
        self.assertEqual(tuple(data["img"].shape), (1, 16, 16, 3))
        self.assertEqual(tuple(data["boxes2d"][0]), (0.5, 0.5, 0.5, 0.5))

    def test_resize_instance_masks(self):
        """Test resize instance masks."""
        data = {
            K.instance_masks: np.array(
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
            ),
            "target_shape": (3, 3),
        }

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

        transfrom = ResizeInstanceMasks(
            in_keys=[K.instance_masks, "target_shape"]
        )
        data_tr = transfrom.apply_to_data([data])[0]
        self.assertTrue((data_tr[K.instance_masks] == expected).all())

    def test_resize_seg_masks(self):
        """Test resize segmentation masks."""
        data = {
            K.seg_masks: np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 1, 2, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0],
                ]
            ),
            "target_shape": (3, 3),
        }
        expected = np.array(
            [
                [0, 0, 0],
                [0, 1, 1],
                [0, 1, 1],
            ]
        )
        transform = ResizeSegMasks(in_keys=[K.seg_masks, "target_shape"])
        data_tr = transform.apply_to_data([data])[0]
        self.assertTrue((data_tr[K.seg_masks] == expected).all())

    def test_resize_intrinsics(self):
        """Test resize intrinsics."""
        data = {
            K.intrinsics: np.array(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            ),
            "scale_factor": (0.5, 0.5),
        }
        expected_intrinsics = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]])
        resize_intrinsics = ResizeIntrinsics(
            in_keys=[K.intrinsics, "scale_factor"]
        )
        self.assertTrue(
            np.allclose(
                resize_intrinsics.apply_to_data([data])[0][K.intrinsics],
                expected_intrinsics,
            )
        )
