# pylint: disable=no-member
"""Test cases for normalize transform."""
import numpy as np

from vis4d.data.const import CommonKeys as K
from vis4d.data.transforms.mask import (
    ConvertInstanceMaskToSegMask,
    RemappingCategories,
)


def test_convert_ins2seg_mask():
    """Test case for convert instance mask to segmentation mask."""
    transform = ConvertInstanceMaskToSegMask()
    data = {
        K.boxes2d_classes: np.array([1, 2, 3], dtype=np.int32),
        K.instance_masks: np.array(
            [
                [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]],
            ],
            dtype=np.uint8,
        ),
    }
    masks = transform.apply_to_data([data])[0][K.seg_masks]
    target = np.array(
        [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 3, 3], [3, 3, 3, 3]],
        dtype=np.uint8,
    )
    assert np.array_equal(masks, target)


def test_remapping_categories():
    """Test case for remapping categories."""
    transform = RemappingCategories([1, 2, 3])
    classes = np.array([1, 2, 3], dtype=np.int32)
    target = np.array([0, 1, 2], dtype=np.int32)
    result = transform([classes])[0]
    assert np.array_equal(result, target)
