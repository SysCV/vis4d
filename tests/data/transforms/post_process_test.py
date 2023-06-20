# pylint: disable=no-member,unexpected-keyword-arg,use-dict-literal
"""Resize transformation tests."""
import numpy as np

from vis4d.data.const import CommonKeys as K
from vis4d.data.transforms.post_process import PostProcessBoxes2D
from vis4d.data.typing import DictData


def test_post_process() -> None:
    """Test resize transformation."""
    data: DictData = {
        K.boxes2d: np.array(
            [
                [10, 10, 20, 20],
                [11, 10, 20, 20],
                [12, 10, 20, 20],
            ],
            dtype=np.float32,
        ),
        K.boxes2d_classes: np.array([0, 0, 0], dtype=np.int32),
        K.input_hw: (128, 128),
    }
    transform = PostProcessBoxes2D(min_area=12 * 12)
    data = transform.apply_to_data([data])[0]

    assert np.all(
        data[K.boxes2d] == np.array([[12, 10, 20, 20]], dtype=np.float32)
    )
