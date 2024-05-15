"""Data connectors for segmentation."""

from vis4d.data.const import CommonKeys as K
from vis4d.engine.connectors import data_key, pred_key

CONN_MASKS_TRAIN = {"images": K.images}

CONN_MASKS_TEST = {"images": K.images, K.original_hw: "original_hw"}

CONN_SEG_LOSS = {
    "output": pred_key("outputs"),
    "target": data_key(K.seg_masks),
}

CONN_MULTI_SEG_LOSS = {
    "outputs": pred_key("outputs"),
    "target": data_key(K.seg_masks),
}

CONN_SEG_EVAL = {
    "prediction": pred_key(K.seg_masks),
    "groundtruth": data_key(K.seg_masks),
}

CONN_SEG_VIS = {
    K.images: data_key(K.images),
    "image_names": data_key(K.sample_names),
    "masks": pred_key("masks"),
}
