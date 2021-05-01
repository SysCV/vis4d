"""Video dataset loader for coco format."""
import logging
from typing import List, Optional

# from scalabel.label.from_coco import coco_to_scalabel
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from scalabel.label.typing import Frame

logger = logging.getLogger(__name__)


def convert_and_load(
    json_path: str, image_root: str, dataset_name: Optional[str] = None
) -> List[Frame]:
    """Convert coco annotations to scalabel format and load them."""
    raise NotImplementedError("Not supported yet")


def register_coco_instances(
    name: str,
    json_path: str,
    image_root: str,
) -> None:  # pragma: no cover
    """Register a dataset in scalabel json annotation format for tracking."""
    # 1. register a function which returns dicts
    DatasetCatalog.register(
        name, lambda: convert_and_load(json_path, image_root, name)
    )

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(json_path=json_path, image_root=image_root)
