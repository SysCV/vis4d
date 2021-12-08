"""Class for processing Scalabel type datasets."""
import random
from typing import Dict, List, Optional

from pytorch_lightning.utilities.distributed import (
    rank_zero_info,
    rank_zero_warn,
)
from scalabel.label.utils import get_leaf_categories
from torch.utils.data import Dataset

from ..common.utils.time import Timer
from ..struct import InputSample

from .datasets import BaseDatasetLoader
from .mapper import build_mapper
from .reference import build_reference_sampler
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
        cats_name2id: Optional[Dict[str, int]] = None,
        image_channel_mode: str = "RGB",
    ):
        """Init."""
        rank_zero_info("Initializing dataset: %s", dataset.cfg.name)
        self.cfg = dataset.cfg
        self.training = training
        self.mapper = build_mapper(
            self.cfg.sample_mapper, training, image_channel_mode
        )

        if cats_name2id is not None:
            discard_labels_outside_set(
                dataset.frames, list(cats_name2id.keys())
            )
        else:
            cats_name2id = {
                v: i
                for i, v in enumerate(
                    [
                        c.name
                        for c in get_leaf_categories(
                            dataset.metadata_cfg.categories
                        )
                    ]
                )
            }
        self.cats_name2id = cats_name2id
        dataset.frames = filter_attributes(
            dataset.frames, dataset.cfg.attributes
        )

        t = Timer()
        frequencies = prepare_labels(
            dataset.frames,
            cats_name2id,
            self.cfg.compute_global_instance_ids,
        )
        rank_zero_info(
            f"Preprocessing {len(dataset.frames)} frames takes {t.time():.2f}"
            " seconds."
        )
        print_class_histogram(frequencies)

        self.dataset = dataset
        self.dataset.frames = DatasetFromList(self.dataset.frames)
        if self.dataset.groups is not None:
            t.reset()
            prepare_labels(
                self.dataset.groups,
                cats_name2id,
                self.cfg.compute_global_instance_ids,
            )
            rank_zero_info(
                f"Preprocessing {len(self.dataset.groups)} groups takes "
                f"{t.time():.2f} seconds."
            )
            self.dataset.groups = DatasetFromList(self.dataset.groups)

        self._fallback_candidates = set(range(len(self.dataset.frames)))
        self.ref_sampler = build_reference_sampler(
            self.cfg.ref_sampler,
            self.dataset.frames,
            self.dataset.groups,
        )
        self.has_sequences = bool(self.ref_sampler.video_to_indices)

    def __len__(self) -> int:
        """Return length of dataset."""
        if self.dataset.groups is not None and not self.training:
            return len(self.dataset.groups)
        return len(self.dataset.frames)

    def __getitem__(self, idx: int) -> List[InputSample]:
        """Fully prepare a sample for training/inference."""
        retry_count = 0
        cur_idx = int(idx)

        if not self.training:
            if self.dataset.groups is not None:
                group = self.dataset.groups[cur_idx]
                if not self.cfg.multi_sensor_inference:
                    cur_data = self.mapper(
                        self.dataset.frames[
                            self.ref_sampler.frame_name_to_idx[group.frames[0]]
                        ]
                    )[0]
                    assert cur_data is not None
                    return [cur_data]

                group_data, group_parameters = self.mapper(group)
                assert group_data is not None
                data = [group_data]
                for fname in group.frames:
                    cur_data, _ = self.mapper(
                        self.dataset.frames[
                            self.ref_sampler.frame_name_to_idx[fname]
                        ],
                        parameters=group_parameters,
                    )
                    assert cur_data is not None
                    data.append(cur_data)
                return data

            cur_data = self.mapper(self.dataset.frames[cur_idx])[0]
            assert cur_data is not None
            data = [cur_data]
            return data

        while True:
            if self.dataset.groups is not None:
                group = self.dataset.groups[
                    self.ref_sampler.frame_to_group[
                        self.ref_sampler.frame_name_to_idx[
                            self.dataset.frames[cur_idx].name
                        ]
                    ]
                ]
                input_data, parameters = self.mapper(
                    self.dataset.frames[cur_idx],
                    group_url=group.url,
                    group_extrinsics=group.extrinsics,
                )
            else:
                input_data, parameters = self.mapper(
                    self.dataset.frames[cur_idx]
                )
            if input_data is not None:
                if input_data.metadata[0].attributes is None:
                    input_data.metadata[0].attributes = {}
                input_data.metadata[0].attributes["keyframe"] = True

                if self.ref_sampler.cfg.num_ref_imgs > 0:
                    ref_data = self.ref_sampler(
                        cur_idx, input_data, self.mapper, parameters
                    )
                    if ref_data is not None:
                        return self.ref_sampler.sort_samples(
                            [input_data] + ref_data
                        )
                else:
                    return [input_data]

            retry_count += 1
            self._fallback_candidates.discard(cur_idx)
            cur_idx = random.sample(self._fallback_candidates, k=1)[0]

            if retry_count >= 5:
                rank_zero_warn(
                    f"Failed to get samples for idx: {cur_idx}, "
                    f"retry count: {retry_count}"
                )
