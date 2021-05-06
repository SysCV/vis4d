"""Function for registering the datasets in detectron2."""


from typing import Dict, List, Optional

from detectron2.data.catalog import DatasetCatalog, MetadataCatalog

from openmt.config import DatasetType

from .coco import convert_and_load_coco
from .mot_challenge import convert_and_load_motchallenge
from .scalabel import load_scalabel


def register_dataset_instances(
    dataset_type: DatasetType,
    name: str,
    json_path: str,
    image_root: str,
    ignore: Optional[List[str]] = None,
    name_mapping: Optional[Dict[str, str]] = None,
) -> None:
    """Register a dataset in scalabel annotation format."""
    if dataset_type == DatasetType.SCALABEL:
        load_func = load_scalabel
    elif dataset_type == DatasetType.COCO:
        load_func = convert_and_load_coco
    elif dataset_type == DatasetType.MOTCHALLENGE:
        load_func = convert_and_load_motchallenge
    else:
        raise NotImplementedError(
            f"Dataset type {dataset_type} currently not supported."
        )

    # 1. register a function which returns List[Frame]
    DatasetCatalog.register(
        name,
        lambda prep_frames=True: load_func(
            json_path, image_root, name, ignore, name_mapping, prep_frames
        ),
    )

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(json_path=json_path, image_root=image_root)
