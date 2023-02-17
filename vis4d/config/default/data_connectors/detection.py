"""Default data connectors for evaluators."""
from vis4d.data.const import CommonKeys as CK
from vis4d.engine.connectors import data_key, pred_key

CONN_BBOX_2D_TRAIN = {
    CK.images: CK.images,
    CK.input_hw: CK.input_hw,
    CK.boxes2d: CK.boxes2d,
    CK.boxes2d_classes: CK.boxes2d_classes,
}

CONN_BBOX_2D_TEST = {
    **CONN_BBOX_2D_TRAIN,
    "original_hw": "original_hw",
}


CONN_RPN_LOSS_2D = {
    "cls_outs": pred_key("rpn.cls"),
    "reg_outs": pred_key("rpn.box"),
    "target_boxes": data_key("boxes2d"),
    "images_hw": data_key("input_hw"),
}

CONN_ROI_LOSS_2D = {
    "class_outs": pred_key("roi.cls_score"),
    "regression_outs": pred_key("roi.bbox_pred"),
    "boxes": pred_key("sampled_proposals.boxes"),
    "boxes_mask": pred_key("sampled_targets.labels"),
    "target_boxes": pred_key("sampled_targets.boxes"),
    "target_classes": pred_key("sampled_targets.classes"),
    "pred_sampled_proposals": pred_key("sampled_proposals"),
}
