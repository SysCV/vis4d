"""Test cases for normalize transform."""
import numpy as np

from vis4d.data.transforms.mask import (
    ConvertInstanceMaskToSegmentationMask,
    RemappingCategories,
)


def test_convert_ins2seg_mask():
    """Test case for convert instance mask to segmentation mask."""
    transform = ConvertInstanceMaskToSegmentationMask()
    classes = np.array([1, 2, 3], dtype=np.int32)
    masks = np.array(
        [
            [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]],
        ],
        dtype=np.uint8,
    )
    target = np.array(
        [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 3, 3], [3, 3, 3, 3]],
        dtype=np.uint8,
    )
    result = transform(classes, masks)
    print(result)
    assert np.array_equal(result, target)


def test_remapping_categories():
    """Test case for remapping categories."""
    transform = RemappingCategories([1, 2, 3])
    classes = np.array([1, 2, 3], dtype=np.int32)
    target = np.array([0, 1, 2], dtype=np.int32)
    result = transform(classes)
    assert np.array_equal(result, target)
