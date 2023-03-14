"""Default data connectors for evaluators."""
from vis4d.data.const import CommonKeys as K
from vis4d.engine.connectors import data_key, pred_key

CONN_IMAGES_TRAIN = {
    K.images: K.images,
    K.input_hw: K.input_hw,
}

CONN_IMAGES_TEST = {
    K.images: K.images,
    K.input_hw: K.input_hw,
    "original_hw": "original_hw",
}

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

CONN_MASK_HEAD_LOSS_2D = {
    "mask_preds": pred_key("masks.mask_pred"),
    "target_masks": data_key("masks"),
    "sampled_target_indices": pred_key("boxes.sampled_target_indices"),
    "sampled_targets": pred_key("boxes.sampled_targets"),
    "sampled_proposals": pred_key("boxes.sampled_proposals"),
}
