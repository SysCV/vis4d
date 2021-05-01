"""Build data loading pipeline for tracking."""
import itertools
from typing import Any, Dict, List, Optional, Union

import torch
from detectron2.config import CfgNode
from detectron2.data import build_batch_data_loader, detection_utils
from detectron2.data.catalog import DatasetCatalog, Metadata, MetadataCatalog
from detectron2.data.common import DatasetFromList
from detectron2.data.samplers import (
    InferenceSampler,
    RepeatFactorTrainingSampler,
    TrainingSampler,
)
from pydantic import BaseModel, validator
from scalabel.label.typing import Frame

from openmt.common.io import DataBackendConfig

from .dataset_mapper import DatasetMapper, MapDataset, ReferenceSamplingConfig
from .samplers import TrackingInferenceSampler
from .utils import (
    filter_empty_annotations,
    identity_batch_collator,
    print_class_histogram,
)


class DataloaderConfig(BaseModel):
    """Config for dataloader."""

    data_backend: DataBackendConfig = DataBackendConfig()
    workers_per_gpu: int
    inference_sampling: str = "sample_based"
    sync_classes_to_intersection: bool = False
    remove_samples_without_labels: bool = False
    train_max_size: Optional[int] = None
    test_max_size: Optional[int] = None
    ref_sampling_cfg: ReferenceSamplingConfig

    @validator("inference_sampling", check_fields=False)
    def validate_inference_sampling(  # pylint: disable=no-self-argument,no-self-use,line-too-long
        cls, value: str
    ) -> str:
        """Check inference_sampling attribute."""
        if value not in ["sample_based", "sequence_based"]:
            raise ValueError(
                "inference_sampling must be sample_based or sequence_based"
            )
        return value


class DataOptions(BaseModel):
    """Options for building the dataloader."""

    dataset: List[Dict[str, Any]]  # type: ignore
    sampler: Optional[Union[TrainingSampler, RepeatFactorTrainingSampler]]
    mapper: DatasetMapper
    total_batch_size: int
    num_workers: int

    class Config:  # needed due to sampler
        """Pydantic configuration for this particular class."""

        arbitrary_types_allowed = True


def update_dataset_to_intersection(
    dataset: List[Frame], cls_intersection: List[str]
) -> None:
    """Update dataset dict using class intersection."""
    for frame in dataset:
        remove_anns = []
        if frame.labels is not None:
            for i, ann in enumerate(frame.labels):
                if ann.category in cls_intersection:
                    assert ann.attributes is not None
                    ann.attributes["category_id"] = cls_intersection.index(
                        ann.category
                    )
                else:
                    remove_anns.append(i)
            for i in reversed(remove_anns):
                frame.labels.pop(i)


def get_dataset_dicts(  # type: ignore
    names: Union[str, List[str]],
    filter_empty: bool = True,
    sync_classes_to_intersection: bool = False,
) -> List[Dict[str, Any]]:
    """Load and prepare dataset dicts."""
    if isinstance(names, str):
        names = [names]
    assert len(names), names
    dataset_frames = [
        DatasetCatalog.get(dataset_name) for dataset_name in names
    ]
    dataset_frames = list(itertools.chain.from_iterable(dataset_frames))

    has_instances = hasattr(dataset_frames[0], "labels")
    if has_instances:
        if sync_classes_to_intersection:
            # synchronize metadata thing_classes, sync idx_to_class_mapping
            metas = [MetadataCatalog.get(d) for d in names]
            classes_per_dataset = [meta.thing_classes for meta in metas]
            # get intersection of classes among all datasets
            intersect_set = set.intersection(*map(set, classes_per_dataset))
            # restore ordering
            class_names = [
                c for c in classes_per_dataset[0] if c in intersect_set
            ]
            assert (
                len(class_names) > 0
            ), f"Classes of datasets {names} have no intersection!"
            for name, meta in zip(names, metas):
                MetadataCatalog.pop(name)
                meta_dict = meta.as_dict()
                meta_dict.update(
                    dict(
                        thing_classes=class_names,
                        idx_to_class_mapping=dict(enumerate(class_names)),
                    )
                )
                MetadataCatalog[name] = Metadata(**meta_dict)

            update_dataset_to_intersection(dataset_frames, class_names)
        else:
            detection_utils.check_metadata_consistency("thing_classes", names)
            class_names = MetadataCatalog.get(names[0]).thing_classes

        overall_frequencies = {c: 0 for c in class_names}
        for name in names:
            meta = MetadataCatalog.get(name)
            for c in class_names:
                overall_frequencies[c] += meta.class_frequencies[c]

        print_class_histogram(overall_frequencies)

    if filter_empty and has_instances:
        dataset_frames = filter_empty_annotations(dataset_frames)

    assert len(dataset_frames) > 0, "No valid data found in {}.".format(
        ",".join(names)
    )
    return dataset_frames


def _train_loader_from_config(
    loader_cfg: DataloaderConfig, cfg: CfgNode
) -> DataOptions:
    """Construct training data loader from config."""
    dataset = get_dataset_dicts(
        cfg.DATASETS.TRAIN,
        loader_cfg.remove_samples_without_labels,
        loader_cfg.sync_classes_to_intersection,
    )
    cfg.INPUT.MAX_SIZE_TRAIN = (
        loader_cfg.train_max_size
        if loader_cfg.train_max_size is not None
        else cfg.INPUT.MAX_SIZE_TRAIN
    )
    mapper = DatasetMapper(loader_cfg.data_backend, cfg)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    if sampler_name == "TrainingSampler":
        sampler = TrainingSampler(len(dataset))
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))

    return DataOptions(
        dataset=dataset,
        sampler=sampler,
        mapper=mapper,
        total_batch_size=cfg.SOLVER.IMS_PER_BATCH,
        num_workers=loader_cfg.workers_per_gpu,
    )


def build_train_loader(
    loader_cfg: DataloaderConfig, det2cfg: CfgNode
) -> torch.utils.data.DataLoader:
    """Build train dataloader with some default features."""
    data_options = _train_loader_from_config(loader_cfg, det2cfg)
    dataset = DatasetFromList(data_options.dataset, copy=False)
    dataset = MapDataset(
        loader_cfg.ref_sampling_cfg, True, dataset, data_options.mapper
    )
    assert isinstance(data_options.sampler, torch.utils.data.sampler.Sampler)
    # aspect_ratio_grouping: tracking datasets usually do not contain
    # sequences with vastly different aspect ratios --> set to False
    return build_batch_data_loader(
        dataset,
        data_options.sampler,
        data_options.total_batch_size,
        aspect_ratio_grouping=False,
        num_workers=data_options.num_workers,
    )


def _test_loader_from_config(
    loader_cfg: DataloaderConfig, cfg: CfgNode, dataset_name: str
) -> DataOptions:
    """Construct testing data loader from config."""
    dataset = get_dataset_dicts(
        dataset_name, False, loader_cfg.sync_classes_to_intersection
    )
    cfg.INPUT.MAX_SIZE_TEST = (
        loader_cfg.test_max_size
        if loader_cfg.test_max_size is not None
        else cfg.INPUT.MAX_SIZE_TEST
    )
    mapper = DatasetMapper(loader_cfg.data_backend, cfg, is_train=False)

    return DataOptions(
        dataset=dataset,
        mapper=mapper,
        total_batch_size=1,
        num_workers=loader_cfg.workers_per_gpu,
    )


def build_test_loader(
    loader_cfg: DataloaderConfig, det2cfg: CfgNode, dataset_name: str
) -> torch.utils.data.DataLoader:
    """Build test dataloader with some default features."""
    data_options = _test_loader_from_config(loader_cfg, det2cfg, dataset_name)
    dataset = DatasetFromList(data_options.dataset, copy=False)
    dataset = MapDataset(
        loader_cfg.ref_sampling_cfg, False, dataset, data_options.mapper
    )
    sampler = (
        TrackingInferenceSampler(dataset)
        if loader_cfg.inference_sampling == "sequence_based"
        else InferenceSampler(len(dataset))
    )
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, 1, drop_last=False
    )
    return torch.utils.data.DataLoader(
        dataset,
        num_workers=data_options.num_workers,
        batch_sampler=batch_sampler,
        collate_fn=identity_batch_collator,
    )
