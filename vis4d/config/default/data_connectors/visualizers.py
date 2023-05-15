"""Default data connectors for visualizers."""
from vis4d.data.const import CommonKeys as K
from vis4d.engine.connectors import data_key, pred_key

CONN_BBOX_2D_VIS = {
    "images": data_key(K.original_images),
    "image_names": data_key(K.sample_names),
    "boxes": pred_key("boxes"),
    "scores": pred_key("scores"),
    "class_ids": pred_key("class_ids"),
}
