"""Default data connectors for visualizers."""
from vis4d.data.const import CommonKeys as CK
from vis4d.engine.connectors import data_key, pred_key

CONN_BBOX_2D_VIS = {
    CK.images: data_key(CK.images),
    "boxes": pred_key("boxes"),
}
