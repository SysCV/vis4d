"""Mask RCNN base model config."""

from vis4d.config.base.models.faster_rcnn import (
    CONN_ROI_LOSS_2D as _CONN_ROI_LOSS_2D,
)
from vis4d.config.base.models.faster_rcnn import (
    CONN_RPN_LOSS_2D as _CONN_RPN_LOSS_2D,
)
from vis4d.data.const import CommonKeys as K
from vis4d.engine.connectors import data_key, pred_key, remap_pred_keys

# Data connectors
CONN_MASK_HEAD_LOSS_2D = {
    "mask_preds": pred_key("masks.mask_pred"),
    "target_masks": data_key(K.instance_masks),
    "sampled_target_indices": pred_key("boxes.sampled_target_indices"),
    "sampled_targets": pred_key("boxes.sampled_targets"),
    "sampled_proposals": pred_key("boxes.sampled_proposals"),
}

CONN_RPN_LOSS_2D = remap_pred_keys(_CONN_RPN_LOSS_2D, "boxes")

CONN_ROI_LOSS_2D = remap_pred_keys(_CONN_ROI_LOSS_2D, "boxes")
