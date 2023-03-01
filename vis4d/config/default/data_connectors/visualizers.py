"""Default data connectors for visualizers."""
from vis4d.data.const import CommonKeys as K
from vis4d.engine.connectors import data_key, pred_key

CONN_BBOX_2D_VIS = {
    K.images: data_key(K.images),
    "boxes": pred_key("boxes"),
}
