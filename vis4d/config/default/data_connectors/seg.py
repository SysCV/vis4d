"""Default data connectors for evaluators."""
from vis4d.data.const import CommonKeys as K
from vis4d.engine.connectors import data_key, pred_key

CONN_MASKS_TRAIN = {K.images: K.images}

CONN_MASKS_TEST = {K.images: K.images, "original_hw": "original_hw"}

CONN_SEG_LOSS = {
    "output": pred_key("outputs"),
    "target": data_key(K.seg_masks),
}

CONN_MULTI_SEG_LOSS = {
    "outputs": pred_key("outputs"),
    "target": data_key(K.seg_masks),
}

CONN_SEG_EVAL = {
    "prediction": pred_key("masks"),
    "groundtruth": data_key(K.seg_masks),
}

CONN_BDD100K_SEG_EVAL = {
    "data_names": data_key("name"),
    "masks_list": pred_key("masks"),
}