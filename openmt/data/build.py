"""Build data loading pipeline for tracking."""
import itertools
import logging
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
from pydantic import BaseModel

from .dataset_mapper import DataloaderConfig, DatasetMapper, MapDataset
from .samplers import TrackingInferenceSampler
from .utils import (
    discard_labels_outside_set,
    filter_empty_annotations,
    identity_batch_collator,
    prepare_labels,
    print_class_histogram,
)


class DataOptions(BaseModel):
    """Options for building the dataloader."""

    dataset: List[Dict[str, Any]]  # type: ignore
    sampler: Optional[Union[TrainingSampler, RepeatFactorTrainingSampler]]
    mapper: DatasetMapper
    total_batch_size: int
    num_workers: int

    class Config:  # needed due to sampler / mapper
        """Pydantic configuration for this particular class."""

        arbitrary_types_allowed = True


def get_dataset_dicts(  # type: ignore
    names: Union[str, List[str]],
    filter_empty: bool = True,
    global_instance_ids: bool = False,
    categories: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Load and prepare dataset dicts."""
    logger = logging.getLogger(__name__)
    if isinstance(names, str):
        names = [names]
    assert len(names), names
    dataset_frames = [
        DatasetCatalog.get(dataset_name) for dataset_name in names
    ]
    dataset_frames = list(itertools.chain.from_iterable(dataset_frames))

    has_instances = False
    for f in dataset_frames:
        if f.labels is not None:
            has_instances = True
            break

    if has_instances:
        if categories is not None:
            logger.info(
                "Filtering categories among the following datasets: %s", names
            )
            logger.info("Given categories: %s", categories)
            # synchronize metadata thing_classes, sync idx_to_class_mapping
            metas = [MetadataCatalog.get(d) for d in names]
            all_categories = list(
                itertools.chain.from_iterable(
                    [meta.thing_classes for meta in metas]
                )
            )
            # get intersection of classes among all datasets
            discard_set = [
                cat for cat in all_categories if cat not in categories
            ]
            # log classes and discarded ones
            if len(discard_set) > 0:
                logger.info(
                    "Discarding the following categories: %s", discard_set
                )
            assert (
                len(categories) > 0
            ), f"Classes of datasets {names} have no intersection!"
            for name, meta in zip(names, metas):
                MetadataCatalog.pop(name)
                meta_dict = meta.as_dict()
                meta_dict.update(
                    dict(
                        thing_classes=categories,
                        idx_to_class_mapping=dict(enumerate(categories)),
                    )
                )
                MetadataCatalog[name] = Metadata(**meta_dict)

            discard_labels_outside_set(dataset_frames, categories)

        # check metadata consistency
        detection_utils.check_metadata_consistency("thing_classes", names)

        if filter_empty:
            dataset_frames = filter_empty_annotations(dataset_frames)

        # add category and instance ids, print class frequencies
        cat_name2id = {
            v: k
            for k, v in MetadataCatalog.get(
                names[0]
            ).idx_to_class_mapping.items()
        }
        frequencies = prepare_labels(
            cat_name2id, dataset_frames, global_instance_ids
        )
        print_class_histogram(frequencies)

    elif categories is not None:
        metas = [MetadataCatalog.get(d) for d in names]
        for name, meta in zip(names, metas):
            MetadataCatalog.pop(name)
            meta_dict = meta.as_dict()
            meta_dict.update(
                dict(
                    thing_classes=categories,
                    idx_to_class_mapping=dict(enumerate(categories)),
                )
            )
            MetadataCatalog[name] = Metadata(**meta_dict)

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
        loader_cfg.compute_global_instance_ids,
        loader_cfg.categories,
    )
    mapper = DatasetMapper(loader_cfg, cfg)

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
        dataset_name, False, False, loader_cfg.categories
    )
    mapper = DatasetMapper(loader_cfg, cfg, is_train=False)

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
