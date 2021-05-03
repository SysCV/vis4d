"""Video dataset loader for coco format."""
import logging
from typing import Dict, List, Optional

# from scalabel.label.from_coco import coco_to_scalabel
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from scalabel.label.typing import Frame

logger = logging.getLogger(__name__)


def convert_and_load(
    json_path: str,
    image_root: str,
    dataset_name: Optional[str] = None,
    ignore_categories: Optional[List[str]] = None,
    name_mapping: Optional[Dict[str, str]] = None,
) -> List[Frame]:
    """Convert coco annotations to scalabel format and load them."""
    raise NotImplementedError("Not supported yet")


def register_coco_instances(
    json_path: str,
    image_root: str,
    name: Optional[str] = None,
    ignore: Optional[List[str]] = None,
    name_mapping: Optional[Dict[str, str]] = None,
) -> None:  # pragma: no cover
    """Conver a coco format dataset to scalabel format and register it."""
    # 1. register a function which returns List[Frame]
    DatasetCatalog.register(
        name,
        lambda: convert_and_load(
            json_path,
            image_root,
            name,
            ignore,
            name_mapping,
        ),
    )

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(json_path=json_path, image_root=image_root)
