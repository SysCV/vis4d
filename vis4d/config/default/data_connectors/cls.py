"""Default data connectors for evaluators."""
from vis4d.data.const import CommonKeys as K
from vis4d.engine.connectors import data_key, pred_key

CONN_CLS_TRAIN = {K.images: K.images}

CONN_CLS_TEST = {K.images: K.images}

CONN_CLS_LOSS = {
    "input": pred_key("logits"),
    "target": data_key("categories"),
}

CONN_CLS_EVAL = {
    "predictions": pred_key("probs"),
    "labels": data_key("categories"),
}
