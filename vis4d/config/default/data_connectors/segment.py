"""Default data connectors for evaluators."""
from vis4d.data.const import CommonKeys as K
from vis4d.engine.connectors import data_key, pred_key

CONN_MASKS_TRAIN = {
    K.images: K.images,
}

CONN_MASKS_TEST = {
    K.images: K.images,
}

CONN_FCN_LOSS = {
    "outputs": pred_key("outputs"),
    "target": data_key(K.segmentation_masks),
}
