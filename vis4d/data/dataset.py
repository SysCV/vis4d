"""Class for processing Scalabel type datasets."""
import random
from typing import List, Optional

from pytorch_lightning.utilities.distributed import (
    rank_zero_info,
    rank_zero_warn,
)
from scalabel.label.utils import get_leaf_categories
from torch.utils.data import Dataset

from ..common.utils.time import Timer
from ..struct import InputSample
from .datasets import BaseDatasetLoader
from .mapper import BaseSampleMapper
from .reference import BaseReferenceSampler
from .utils import (
    DatasetFromList,
    discard_labels_outside_set,
    filter_attributes,
    prepare_labels,
    print_class_histogram,
)


class ScalabelDataset(Dataset):  # type: ignore
    """Preprocess Scalabel format data into Vis4D input format."""

    def __init__(
        self,
        dataset: BaseDatasetLoader,
        training: bool,
        mapper: Optional[BaseSampleMapper] = None,
        ref_sampler: Optional[BaseReferenceSampler] = None,
    ):
        """Init."""
        rank_zero_info("Initializing dataset: %s", dataset.name)
        self.training = training
        self.mapper = mapper if mapper is not None else BaseSampleMapper()

        if len(self.mapper.cats_name2id) > 0:
            class_list = list(
                set(
                    cls
                    for field in self.mapper.cats_name2id
                    for cls in list(self.mapper.cats_name2id[field].keys())
                )
            )
            discard_labels_outside_set(dataset.frames, class_list)
        else:
            class_list = list(
                set(
                    c.name
                    for c in get_leaf_categories(
                        dataset.metadata_cfg.categories
                    )
                )
            )

        dataset.frames = filter_attributes(dataset.frames, dataset.attributes)
        cmpt_gbl_ids = dataset.compute_global_instance_ids

        t = Timer()
        frequencies = prepare_labels(dataset.frames, class_list, cmpt_gbl_ids)
        rank_zero_info(
            f"Preprocessing {len(dataset.frames)} frames takes {t.time():.2f}"
            " seconds."
        )
        print_class_histogram(frequencies)

        self.dataset = dataset
        self.dataset.frames = DatasetFromList(self.dataset.frames)
        if self.dataset.groups is not None:
            t.reset()
            prepare_labels(self.dataset.groups, class_list, cmpt_gbl_ids)
            rank_zero_info(
                f"Preprocessing {len(self.dataset.groups)} groups takes "
                f"{t.time():.2f} seconds."
            )
            self.dataset.groups = DatasetFromList(self.dataset.groups)

        self._fallback_candidates = set(range(len(self.dataset.frames)))

        self.ref_sampler = (
            ref_sampler if ref_sampler is not None else BaseReferenceSampler()
        )
        self.ref_sampler.create_mappings(
            self.dataset.frames, self.dataset.groups
        )

        self.has_sequences = bool(self.ref_sampler.video_to_indices)
        self._show_retry_warn = True

    def __len__(self) -> int:
        """Return length of dataset."""
        if self.dataset.groups is not None and not self.training:
            return len(self.dataset.groups)
        return len(self.dataset.frames)

    def __getitem__(self, idx: int) -> List[InputSample]:
        """Fully prepare a sample for training/inference."""
        retry_count = 0
        cur_idx = int(idx)

        frame2id = self.ref_sampler.frame_name_to_idx
        if not self.training:
            if self.dataset.groups is not None:
                group = self.dataset.groups[cur_idx]
                if not self.dataset.multi_sensor_inference:
                    cur_data = self.mapper(
                        self.dataset.frames[frame2id[group.frames[0]]],
                        self.training,
                    )
                    assert cur_data is not None
                    return [cur_data]

                group_data = self.mapper(group, self.training)
                assert group_data is not None
                data = [group_data]
                for fname in group.frames:
                    cur_data = self.mapper(
                        self.dataset.frames[frame2id[fname]], self.training
                    )
                    assert cur_data is not None
                    data.append(cur_data)
                return data

            cur_data = self.mapper(self.dataset.frames[cur_idx], self.training)
            assert cur_data is not None
            data = [cur_data]
            return data

        while True:
            cur_frame = self.dataset.frames[cur_idx]
            if self.dataset.groups is not None:
                group = self.dataset.groups[
                    self.ref_sampler.frame_to_group[
                        self.ref_sampler.frame_name_to_idx[cur_frame.name]
                    ]
                ]
                input_data = self.mapper(
                    cur_frame,
                    self.training,
                    group_url=group.url,
                    group_extrinsics=group.extrinsics,
                )
            else:
                input_data = self.mapper(cur_frame, self.training)
            if input_data is not None:
                if input_data.metadata[0].attributes is None:
                    input_data.metadata[0].attributes = {}
                input_data.metadata[0].attributes["keyframe"] = True

                if self.ref_sampler.num_ref_imgs > 0:
                    ref_data = self.ref_sampler(
                        cur_idx, input_data, self.training, self.mapper
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
