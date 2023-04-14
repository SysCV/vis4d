"""Default data connectors for detection."""
from vis4d.data.const import CommonKeys as K
from vis4d.engine.connectors import data_key, pred_key

CONN_BBOX_2D_TRAIN = {
    K.images: K.images,
    K.input_hw: K.input_hw,
    K.boxes2d: K.boxes2d,
    K.boxes2d_classes: K.boxes2d_classes,
}

CONN_BBOX_2D_TEST = {
    **CONN_BBOX_2D_TRAIN,
    "original_hw": "original_hw",
}

CONN_BOX_LOSS_2D = {
    "cls_outs": pred_key("cls_score"),
    "reg_outs": pred_key("bbox_pred"),
    "target_boxes": data_key("boxes2d"),
    "images_hw": data_key("input_hw"),
}
