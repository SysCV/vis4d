"""Function for registering the datasets in detectron2."""
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog

from openmt.config import Dataset, DatasetType

from .bdd100k import convert_and_load_bdd100k
from .coco import convert_and_load_coco
from .custom import convert_and_load_directory
from .motchallenge import convert_and_load_motchallenge
from .scalabel import load_scalabel


def register_dataset_instances(dataset: Dataset) -> None:
    """Register a dataset in scalabel annotation format."""
    if dataset.type == DatasetType.SCALABEL:
        load_func = load_scalabel
    elif dataset.type == DatasetType.BDD100K:
        load_func = convert_and_load_bdd100k
    elif dataset.type == DatasetType.COCO:
        load_func = convert_and_load_coco
    elif dataset.type == DatasetType.CUSTOM:
        load_func = convert_and_load_directory
    elif dataset.type == DatasetType.MOTCHALLENGE:
        load_func = convert_and_load_motchallenge
    else:
        raise NotImplementedError(
            f"Dataset type {dataset.type} currently not supported."
        )

    # 1. register a function which returns List[Frame]
    DatasetCatalog.register(
        dataset.name,
        lambda prep_frames=True: load_func(
            dataset.data_root,
            dataset.annotations,
            dataset.name,
            dataset.config_path,
            prep_frames,
        ),
    )

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(dataset.name).set(
        annotations=dataset.annotations,
        data_root=dataset.data_root,
        cfg_path=dataset.config_path,
    )
