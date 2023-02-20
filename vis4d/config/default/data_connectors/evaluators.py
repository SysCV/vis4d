"""Default data connectors for evaluators."""
from vis4d.engine.connectors import data_key, pred_key

CONN_COCO_BBOX_EVAL = {
    "coco_image_id": data_key("coco_image_id"),
    "pred_boxes": pred_key("boxes"),
    "pred_scores": pred_key("scores"),
    "pred_classes": pred_key("class_ids"),
}
