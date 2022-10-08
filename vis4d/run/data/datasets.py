"""Default dataset configs used in Vis4D."""
from vis4d.data.datasets import COCO
from vis4d.data.io.base import DataBackend
from vis4d.eval import COCOEvaluator


def coco_train(data_backend: DataBackend) -> COCO:
    """Create COCO train dataset with default data root."""
    return COCO("data/COCO/", data_backend=data_backend)


def coco_val(data_backend: DataBackend) -> COCO:
    """Create COCO val dataset with default data root."""
    return COCO("data/COCO/", split="val2017", data_backend=data_backend)


def coco_val_eval() -> COCOEvaluator:
    """Create COCO val evaluator with default data root."""
    return COCOEvaluator("data/COCO/", split="val2017")
