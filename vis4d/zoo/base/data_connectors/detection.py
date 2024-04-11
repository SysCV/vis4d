"""Data connectors for detection."""

from vis4d.data.const import CommonKeys as K
from vis4d.engine.connectors import data_key, pred_key

CONN_BBOX_2D_TRAIN = {
    "images": K.images,
    "input_hw": K.input_hw,
    "boxes2d": K.boxes2d,
    "boxes2d_classes": K.boxes2d_classes,
}

CONN_BBOX_2D_TEST = {
    "images": K.images,
    "input_hw": K.input_hw,
    "original_hw": K.original_hw,
}

CONN_BOX_LOSS_2D = {
    "cls_outs": pred_key("cls_score"),
    "reg_outs": pred_key("bbox_pred"),
    "target_boxes": data_key(K.boxes2d),
    "images_hw": data_key(K.input_hw),
}
