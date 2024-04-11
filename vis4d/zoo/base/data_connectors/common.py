"""Data connectors for common tasks."""

from vis4d.data.const import CommonKeys as K

CONN_IMAGES_TRAIN = {"images": K.images, "input_hw": K.input_hw}

CONN_IMAGES_TEST = {
    "images": K.images,
    "input_hw": K.input_hw,
    "original_hw": K.original_hw,
}
