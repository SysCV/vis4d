"""Test Flip transform."""
import copy

import numpy as np

from vis4d.data.transforms.flip import FlipBoxes2D, FlipImage


def test_flip_image():
    """Test the FlipImage transform."""
    images = np.random.rand(1, 16, 16, 3)

    transform = FlipImage(direction="horizontal")
    images_tr = transform(copy.deepcopy(images))
    assert images_tr.shape == (1, 16, 16, 3)
    assert (images_tr == images[:, :, ::-1, :]).all()

    transform = FlipImage(direction="vertical")
    images_tr = transform(copy.deepcopy(images))
    assert images_tr.shape == (1, 16, 16, 3)
    assert (images_tr == images[:, ::-1, :, :]).all()


def test_flip_boxes2d():
    """Test the FlipBoxes2D transform."""
    images = np.random.rand(1, 16, 16, 3)
    boxes = np.random.rand(3, 4)

    transform = FlipBoxes2D(direction="horizontal")
    boxes_tr = transform(copy.deepcopy(boxes), copy.deepcopy(images))
    assert boxes_tr.shape == (3, 4)
    assert (boxes_tr[:, 1::2] == boxes[:, 1::2]).all()
    assert (boxes_tr[:, 0::2] == np.flip(16 - boxes[:, 0::2], 1)).all()

    transform = FlipBoxes2D(direction="vertical")
    boxes_tr = transform(copy.deepcopy(boxes), copy.deepcopy(images))
    assert boxes_tr.shape == (3, 4)
    assert (boxes_tr[:, 0::2] == boxes[:, 0::2]).all()
    assert (boxes_tr[:, 1::2] == np.flip(16 - boxes[:, 1::2], 1)).all()
