"""Vis4D base dataset class."""
import abc
import copy
import os
import pickle
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from pytorch_lightning.utilities.rank_zero import (
    rank_zero_info,
    rank_zero_warn,
)
from scalabel.label.utils import get_leaf_categories
from torch.utils.data import Dataset

from vis4d.common.registry import RegistryHolder
from vis4d.common.utils.time import Timer
from vis4d.struct import MetricLogs

from ..common.io import BaseDataBackend, FileBackend
from ..common.registry import RegistryHolder
from ..reference import BaseReferenceSampler
from ..struct import (
    ALLOWED_INPUTS,
    ALLOWED_TARGETS,
    Boxes2D,
    Boxes3D,
    CategoryMap,
    Extrinsics,
    Images,
    InputSample,
    InstanceMasks,
    Intrinsics,
    PointCloud,
    SemanticMasks,
)
from ..utils import (
    DatasetFromList,
    discard_labels_outside_set,
    filter_attributes,
    prepare_labels,
    print_class_histogram,
)

# TODO make abstract


class BaseDataset(metaclass=RegistryHolder):
    """Interface for loading dataset to Scalabel format."""

    def __init__(
        self,
        training: bool,
        name: str,
        data_root: str,
        annotations: Optional[str] = None,
        config_path: Optional[str] = None,
        eval_metrics: Optional[List[str]] = None,
        ignore_unknown_cats: bool = False,
        cache_as_binary: bool = False,
        num_processes: int = 4,
        collect_device: str = "cpu",
        compute_global_instance_ids: bool = False,
        ref_sampler: Optional[BaseReferenceSampler] = None,
        data_backend: BaseDataBackend = FileBackend(),
        inputs_to_load: Tuple[str, ...] = ("images",),
        targets_to_load: Tuple[str, ...] = ("boxes2d",),
        skip_empty_samples: bool = False,
        bg_as_class: bool = False,
        image_backend: str = "PIL",
        image_channel_mode: str = "RGB",
        category_map: Optional[CategoryMap] = None,
    ):
        """Init dataset loader."""
        super().__init__()
        rank_zero_info("Initializing dataset: %s", name)
        self.training = training

        self.image_backend = image_backend
        self.inputs_to_load = inputs_to_load
        if not all(ele in ALLOWED_INPUTS for ele in inputs_to_load):
            raise ValueError(
                f"Found invalid inputs: {inputs_to_load}, "
                f"allowed set of inputs: {ALLOWED_INPUTS}"
            )
        self.targets_to_load = targets_to_load
        if not all(ele in ALLOWED_TARGETS for ele in targets_to_load):
            raise ValueError(
                f"Found invalid targets: {targets_to_load}, "
                f"allowed set of targets: {ALLOWED_TARGETS}"
            )
        self.bg_as_class = bg_as_class
        self.skip_empty_samples = skip_empty_samples
        self.image_channel_mode = image_channel_mode

        self.data_backend = data_backend
        self.cats_name2id: Dict[str, Dict[str, int]] = {}
        if category_map is not None:
            self.setup_categories(category_map)

        self.name = name
        self.data_root = data_root
        self.annotations = annotations
        self.config_path = config_path
        self.eval_metrics = eval_metrics
        self.ignore_unknown_cats = ignore_unknown_cats
        self.collect_device = collect_device
        self.cache_as_binary = cache_as_binary
        self.compute_global_instance_ids = compute_global_instance_ids
        self.num_processes = num_processes
        self.custom_save = custom_save

        if self.eval_metrics is None:
            self.eval_metrics = []
        self._check_metrics()

        timer = Timer()
        if cache_as_binary:
            dataset, has_sequences = self.load_cached_dataset()
        else:
            dataset, has_sequences = self.load_dataset()

        self.dataset = DatasetFromList(dataset)
        self.has_sequences = has_sequences

        rank_zero_info(f"Loading {name} takes {timer.time():.2f} seconds.")

        self.ref_sampler = (
            ref_sampler if ref_sampler is not None else BaseReferenceSampler()
        )
        self.ref_sampler.create_mappings(
            self.dataset.frames, self.dataset.groups
        )

        self._fallback_candidates = set(range(len(self.dataset)))
        self._show_retry_warn = True

    def load_cached_dataset(self) -> Dataset:
        """Load cached dataset from file."""
        if self.annotations is None:  # pragma: no cover
            cache_path = self.data_root.rstrip("/") + ".pkl"
        else:
            cache_path = self.annotations.rstrip("/") + ".pkl"
        if not os.path.exists(cache_path):
            dataset = self.load_dataset()
            with open(cache_path, "wb") as file:
                file.write(pickle.dumps(dataset))
        else:
            with open(cache_path, "rb") as file:
                dataset = pickle.loads(file.read())
        return dataset

    @abc.abstractmethod
    def load_dataset(self) -> List[DataDict]:
        """Load and possibly convert dataset to Scalabel format."""
        raise NotImplementedError

    def _check_metrics(self) -> None:
        """Check if evaluation metrics specified are valid."""
        assert self.eval_metrics is not None
        for metric in self.eval_metrics:
            if metric not in _eval_mapping:  # pragma: no cover
                raise KeyError(
                    f"metric {metric} is not supported in"
                    f" dataset {self.name}"
                )

    def evaluate(
        self, metric: str, predictions: List[Frame], gts: List[Frame]
    ) -> Tuple[MetricLogs, str]:
        """Convert predictions from Scalabel format and evaluate.

        Returns a dictionary of scores to log and a pretty printed string.
        """
        raise NotImplementedError

    def save_predictions(
        self, output_dir: str, metric: str, predictions: List[Frame]
    ) -> None:
        """Save model predictions in Scalabel format."""
        raise NotImplementedError

    def load_sample(
        self,
        sample: DataDict,
        training: bool,
    ) -> Optional[InputSample]:
        raise NotImplementedError

    def __len__(self) -> int:
        """Return length of dataset."""
        if self.inference_with_group and self.dataset.groups is not None:
            return len(self.dataset.groups)
        return len(self.dataset.frames)

    def __getitem__(self, idx: int) -> List[InputData]:
        """Fully prepare a sample for training/inference."""
        retry_count = 0
        cur_idx = int(idx)

        if not self.training:
            cur_data = self.load_sample(self.dataset[cur_idx], self.training)
            assert cur_data is not None
            data = [cur_data]
            return data

        while True:
            cur_frame = self.dataset.frames[cur_idx]
            input_data = self.load_sample(cur_frame, self.training)
            if input_data is not None:
                if input_data.metadata[0].attributes is None:
                    input_data.metadata[0].attributes = {}
                input_data.metadata[0].attributes["keyframe"] = True

                if self.ref_sampler.num_ref_imgs > 0:
                    ref_data = self.ref_sampler(
                        cur_idx, input_data, self.training, self.load_sample
                    )
                    if ref_data is not None:
                        return [input_data] + ref_data
                else:
                    return [input_data]

            retry_count += 1
            self._fallback_candidates.discard(cur_idx)
            cur_idx = random.sample(self._fallback_candidates, k=1)[0]

            if self._show_retry_warn and retry_count >= 5:
                rank_zero_warn(
                    f"Failed to get an input sample for idx {cur_idx} after "
                    f"{retry_count} retries, this happens e.g. when "
                    "skip_empty_samples is activated and there are many "
                    "samples without (valid) labels. Please check your class "
                    "configuration and/or dataset labels if this is "
                    "undesired behavior."
                )
                self._show_retry_warn = False
