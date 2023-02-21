"""Default data connectors for evaluators."""
from vis4d.data.const import CommonKeys as CK
from vis4d.engine.connectors import data_key, pred_key

CONN_MASKS_TRAIN = {
    CK.images: CK.images,
}

CONN_MASKS_TEST = {
    CK.images: CK.images,
}

CONN_FCN_LOSS = {
    "outputs": pred_key("outputs"),
    "target": data_key(CK.segmentation_masks),
}
