"""Class for processing Scalabel type datasets."""
import copy
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from pytorch_lightning.utilities.distributed import (
    rank_zero_info,
    rank_zero_warn,
)
from scalabel.label.typing import Extrinsics as ScalabelExtrinsics
from scalabel.label.typing import Frame, FrameGroup, ImageSize
from scalabel.label.typing import Intrinsics as ScalabelIntrinsics
from scalabel.label.typing import Label
from scalabel.label.utils import (
    check_crowd,
    check_ignored,
    get_leaf_categories,
    get_matrix_from_extrinsics,
    get_matrix_from_intrinsics,
)
from torch.utils.data import Dataset

from ..common.io import build_data_backend
from ..common.utils.time import Timer
from ..struct import (
    Boxes2D,
    Boxes3D,
    Extrinsics,
    Images,
    InputSample,
    InstanceMasks,
    Intrinsics,
    LabelInstances,
    SemanticMasks,
)
from .datasets import BaseDatasetLoader
from .transforms import AugParams, build_augmentations
from .utils import (
    DatasetFromList,
    discard_labels_outside_set,
    filter_attributes,
    im_decode,
    prepare_labels,
    print_class_histogram,
)

__all__ = ["ScalabelDataset"]


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
        self.image_channel_mode = image_channel_mode
        self.sampling_cfg = self.cfg.dataloader.ref_sampling
        self.data_backend = build_data_backend(
            self.cfg.dataloader.data_backend
        )
        rank_zero_info(
            "Using data backend: %s", self.cfg.dataloader.data_backend.type
        )

        self.transformations = build_augmentations(
            self.cfg.dataloader.transformations
        )
        rank_zero_info("Transformations used: %s", self.transformations)

        fields_to_load = self.cfg.dataloader.fields_to_load
        allowed_files = [
            "boxes2d",
            "boxes3d",
            "instance_masks",
            "semantic_masks",
            "intrinsics",
            "extrinsics",
        ]
        for field in fields_to_load:
            assert (
                field in allowed_files
            ), f"Unrecognized field={field}, allowed fields={allowed_files}"
        assert (
            not "instance_masks" in fields_to_load
            or not "semantic_masks" in fields_to_load
        ), (
            "Both instance_masks and semantic_masks are specified, "
            "but only one should be."
        )
        self.training = training

        if self.cfg.dataloader.skip_empty_samples and not self.training:
            rank_zero_warn(  # pragma: no cover
                f"'skip_empty_samples' activated for dataset {self.cfg.name}"
                "in test mode. This option is only available in training."
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
            self.cfg.dataloader.compute_global_instance_ids,
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
                self.cfg.dataloader.compute_global_instance_ids,
            )
            rank_zero_info(
                f"Preprocessing {len(self.dataset.groups)} groups takes "
                f"{t.time():.2f} seconds."
            )
            self.dataset.groups = DatasetFromList(self.dataset.groups)

        self._fallback_candidates = set(range(len(self.dataset.frames)))
        self.video_to_indices: Dict[str, List[int]] = defaultdict(list)
        self._create_video_mapping()
        self.has_sequences = bool(self.video_to_indices)

        if self.dataset.groups is not None:
            self.frame_name_to_idx = {
                f.name: i for i, f in enumerate(self.dataset.frames)
            }
            self.frame_to_group: Dict[int, int] = {}
            self.frame_to_sensor_id: Dict[int, int] = {}
            for i, g in enumerate(self.dataset.groups):
                for sensor_id, fname in enumerate(g.frames):
                    self.frame_to_group[self.frame_name_to_idx[fname]] = i
                    self.frame_to_sensor_id[
                        self.frame_name_to_idx[fname]
                    ] = sensor_id

    def __len__(self) -> int:
        """Return length of dataset."""
        if self.dataset.groups is not None and not self.training:
            return len(self.dataset.groups)
        return len(self.dataset.frames)

    def _create_video_mapping(self) -> None:
        """Creating mapping from video id to frame / group indices."""
        video_to_frameidx: Dict[str, List[int]] = defaultdict(list)
        if self.dataset.groups is not None:
            for idx, group in enumerate(self.dataset.groups):
                if group.videoName is not None:
                    assert (
                        group.frameIndex is not None
                    ), "found videoName but no frameIndex!"
                    video_to_frameidx[group.videoName].append(group.frameIndex)
                    self.video_to_indices[group.videoName].append(idx)
        else:
            for idx, frame in enumerate(self.dataset.frames):
                if frame.videoName is not None:
                    assert (
                        frame.frameIndex is not None
                    ), "found videoName but no frameIndex!"
                    video_to_frameidx[frame.videoName].append(frame.frameIndex)
                    self.video_to_indices[frame.videoName].append(idx)

        # sort dataset indices by frame indices
        for key, idcs in self.video_to_indices.items():
            zip_frame_idx = sorted(zip(video_to_frameidx[key], idcs))
            self.video_to_indices[key] = [idx for _, idx in zip_frame_idx]

    def sample_ref_indices(
        self, video: str, key_dataset_index: int
    ) -> List[int]:
        """Sample reference dataset indices given video and keyframe index."""
        dataset_indices = self.video_to_indices[video]
        sensor_id: Optional[int] = None
        if self.dataset.groups is not None:
            sensor_id = self.frame_to_sensor_id[key_dataset_index]
            key_dataset_index = self.frame_to_group[key_dataset_index]

        key_index = dataset_indices.index(key_dataset_index)

        if self.sampling_cfg.type == "uniform":
            left = max(0, key_index - self.sampling_cfg.scope)
            right = min(
                key_index + self.sampling_cfg.scope, len(dataset_indices) - 1
            )
            valid_inds = (
                dataset_indices[left:key_index]
                + dataset_indices[key_index + 1 : right + 1]
            )
            ref_dataset_indices: List[int] = np.random.choice(
                valid_inds, self.sampling_cfg.num_ref_imgs, replace=False
            ).tolist()
        elif self.sampling_cfg.type == "sequential":
            right = key_index + 1 + self.sampling_cfg.num_ref_imgs
            if right <= len(dataset_indices):
                ref_dataset_indices = dataset_indices[key_index + 1 : right]
            else:
                left = key_index - (right - len(dataset_indices))
                ref_dataset_indices = (
                    dataset_indices[left:key_index]
                    + dataset_indices[key_index + 1 :]
                )
        else:
            raise NotImplementedError(
                f"Reference view sampling {self.sampling_cfg.type} not "
                f"implemented."
            )

        if self.dataset.groups is not None and sensor_id is not None:
            for i, ref_id in enumerate(ref_dataset_indices):
                fname = self.dataset.groups[ref_id].frames[sensor_id]
                ref_dataset_indices[i] = self.frame_name_to_idx[fname]

        return ref_dataset_indices

    def sort_samples(
        self, input_samples: List[InputSample]
    ) -> List[InputSample]:
        """Sort samples according to sampling cfg."""
        if self.sampling_cfg.frame_order == "key_first":
            return input_samples
        if self.sampling_cfg.frame_order == "temporal":
            return sorted(
                input_samples,
                key=lambda x: x.metadata[0].frameIndex
                if x.metadata[0].frameIndex is not None
                else 0,
            )
        raise NotImplementedError(
            f"Frame ordering {self.sampling_cfg.frame_order} not "
            f"implemented."
        )

    def sample_ref_views(
        self,
        cur_idx: int,
        key_data: InputSample,
        parameters: Optional[List[AugParams]],
        num_retry: int = 3,
    ) -> Optional[List[InputSample]]:
        """Sample reference views from key view."""
        vid_id = key_data.metadata[0].videoName
        for _ in range(num_retry):
            if vid_id is not None:
                ref_data = []
                for ref_idx in self.sample_ref_indices(vid_id, cur_idx):
                    ref_sample = self.get_sample(
                        self.dataset.frames[ref_idx],
                        parameters=parameters,
                    )[0]
                    if ref_sample is None:
                        break  # pragma: no cover
                    ref_data.append(ref_sample)
            else:  # pragma: no cover
                if parameters is not None:
                    ref_data = [
                        key_data for _ in range(self.sampling_cfg.num_ref_imgs)
                    ]
                else:
                    ref_data = []
                    for _ in range(self.sampling_cfg.num_ref_imgs):
                        ref_sample = self.get_sample(
                            self.dataset.frames[cur_idx]
                        )[0]
                        if ref_sample is None:
                            break
                        ref_data.append(ref_sample)
            if (
                not self.sampling_cfg.skip_nomatch_samples
                or self.has_matches(key_data, ref_data)
            ) and self.sampling_cfg.num_ref_imgs == len(ref_data):
                return ref_data
        return None

    @staticmethod
    def has_matches(
        key_data: InputSample, ref_data: List[InputSample]
    ) -> bool:
        """Check if key / ref data have matches."""
        key_track_ids = key_data.targets.boxes2d[0].track_ids
        for ref_view in ref_data:
            ref_track_ids = ref_view.targets.boxes2d[0].track_ids
            match = key_track_ids.view(-1, 1) == ref_track_ids.view(1, -1)
            if match.any():
                return True
        return False  # pragma: no cover

    def __getitem__(self, idx: int) -> List[InputSample]:
        """Fully prepare a sample for training/inference."""
        retry_count = 0
        cur_idx = int(idx)

        if not self.training:
            if self.dataset.groups is not None:
                group = self.dataset.groups[cur_idx]
                if not self.cfg.multi_sensor_inference:
                    cur_data = self.get_sample(
                        self.dataset.frames[
                            self.frame_name_to_idx[group.frames[0]]
                        ]
                    )[0]
                    assert cur_data is not None
                    return [cur_data]

                group_data, group_parameters = self.get_sample(group)
                assert group_data is not None
                data = [group_data]
                for fname in group.frames:
                    cur_data, _ = self.get_sample(
                        self.dataset.frames[self.frame_name_to_idx[fname]],
                        parameters=group_parameters,
                    )
                    assert cur_data is not None
                    data.append(cur_data)
                return data

            cur_data = self.get_sample(self.dataset.frames[cur_idx])[0]
            assert cur_data is not None
            data = [cur_data]
            return data

        while True:
            input_data, parameters = self.get_sample(
                self.dataset.frames[cur_idx]
            )
            if input_data is not None:
                if input_data.metadata[0].attributes is None:
                    input_data.metadata[0].attributes = {}
                input_data.metadata[0].attributes["keyframe"] = True

                if self.sampling_cfg.num_ref_imgs > 0:
                    ref_data = self.sample_ref_views(
                        cur_idx, input_data, parameters
                    )
                    if ref_data is not None:
                        return self.sort_samples([input_data] + ref_data)
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

    def load_input(
        self, sample: Frame, use_empty: Optional[bool] = False
    ) -> InputSample:
        """Load image according to data_backend."""
        if not use_empty:
            assert sample.url is not None
            im_bytes = self.data_backend.get(sample.url)
            image = im_decode(im_bytes, mode=self.image_channel_mode)
        else:
            image = np.empty((128, 128, 3), dtype=np.uint8)

        sample.size = ImageSize(width=image.shape[1], height=image.shape[0])
        image = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1)),
            dtype=torch.float32,
        ).unsqueeze(0)
        images = Images(image, [(image.shape[3], image.shape[2])])
        input_data = InputSample([copy.deepcopy(sample)], images)

        if (
            sample.intrinsics is not None
            and "intrinsics" in self.cfg.dataloader.fields_to_load
        ):
            input_data.intrinsics = self.load_intrinsics(sample.intrinsics)

        if (
            sample.extrinsics is not None
            and "extrinsics" in self.cfg.dataloader.fields_to_load
        ):
            input_data.extrinsics = self.load_extrinsics(sample.extrinsics)
        return input_data

    def load_annotation(
        self,
        sample: InputSample,
        labels: Optional[List[Label]],
    ) -> None:
        """Transform annotations."""
        labels_used = []
        if labels is not None:
            category_dict = {}
            instance_id_dict = {}
            for label in labels:
                assert label.attributes is not None
                assert label.category is not None
                if not check_crowd(label) and not check_ignored(label):
                    labels_used.append(label)
                    if label.category not in category_dict:
                        category_dict[label.category] = int(
                            label.attributes["category_id"]
                        )
                    if label.id not in instance_id_dict:
                        instance_id_dict[label.id] = int(
                            label.attributes["instance_id"]
                        )

            if labels_used:
                if "instance_masks" in self.cfg.dataloader.fields_to_load:
                    instance_masks = InstanceMasks.from_scalabel(
                        labels_used,
                        category_dict,
                        instance_id_dict,
                        sample.metadata[0].size,
                    )
                    sample.targets.instance_masks = [instance_masks]

                if "semantic_masks" in self.cfg.dataloader.fields_to_load:
                    semantic_masks = SemanticMasks.from_scalabel(
                        labels_used,
                        category_dict,
                        instance_id_dict,
                        sample.metadata[0].size,
                    )
                    sample.targets.semantic_masks = [semantic_masks]

                if "boxes2d" in self.cfg.dataloader.fields_to_load:
                    boxes2d = Boxes2D.from_scalabel(
                        labels_used, category_dict, instance_id_dict
                    )
                    if (
                        len(boxes2d) == 0
                        and len(sample.targets.instance_masks[0]) > 0
                    ):  # pragma: no cover
                        boxes2d = sample.targets.instance_masks[
                            0
                        ].get_boxes2d()
                    sample.targets.boxes2d = [boxes2d]

                if "boxes3d" in self.cfg.dataloader.fields_to_load:
                    boxes3d = Boxes3D.from_scalabel(
                        labels_used, category_dict, instance_id_dict
                    )
                    sample.targets.boxes3d = [boxes3d]

    def transform_input(
        self,
        sample: InputSample,
        parameters: Optional[List[AugParams]] = None,
    ) -> List[AugParams]:
        """Apply transforms to input sample."""
        if parameters is None:
            parameters = []
        else:
            assert len(parameters) == len(self.transformations), (
                "Length of augmentation parameters must equal the number of "
                "augmentations!"
            )
        for i, aug in enumerate(self.transformations):
            if len(parameters) < len(self.transformations):
                parameters.append(aug.generate_parameters(sample))
            sample, _ = aug(sample, parameters[i])
        return parameters

    def postprocess_annotation(
        self, im_wh: Tuple[int, int], targets: LabelInstances
    ) -> None:
        """Process annotations after transform."""
        if len(targets.boxes2d[0]) == 0:
            return
        if self.cfg.dataloader.clip_bboxes_to_image:
            targets.boxes2d[0].clip(im_wh)
        keep = targets.boxes2d[0].area >= self.cfg.dataloader.min_bboxes_area
        targets.boxes2d = [targets.boxes2d[0][keep]]
        if len(targets.boxes3d[0]) > 0:
            targets.boxes3d = [targets.boxes3d[0][keep]]
        if len(targets.instance_masks[0]) > 0:
            targets.instance_masks = [targets.instance_masks[0][keep]]

    @staticmethod
    def load_intrinsics(intrinsics: ScalabelIntrinsics) -> Intrinsics:
        """Transform intrinsic camera matrix according to augmentations."""
        intrinsic_matrix = torch.from_numpy(
            get_matrix_from_intrinsics(intrinsics)
        ).to(torch.float32)
        return Intrinsics(intrinsic_matrix)

    @staticmethod
    def load_extrinsics(extrinsics: ScalabelExtrinsics) -> Extrinsics:
        """Transform extrinsics from Scalabel to Vis4D."""
        extrinsics_matrix = torch.from_numpy(
            get_matrix_from_extrinsics(extrinsics)
        ).to(torch.float32)
        return Extrinsics(extrinsics_matrix)

    def get_sample(
        self,
        sample: Frame,
        parameters: Optional[List[AugParams]] = None,
    ) -> Tuple[Optional[InputSample], Optional[List[AugParams]]]:
        """Prepare a single sample in detect format.

        Args:
            sample (Frame): Metadata of one image, in scalabel format.
            Serialized as dict due to multi-processing.
            parameters (List[AugParams]): Augmentation parameter list.

        Returns:
            InputSample: Data format that the model accepts.
            List[AugParams]: augmentation parameters, s.t. ref views can be
            augmented with the same parameters.
        """
        if (
            self.cfg.dataloader.skip_empty_samples
            and (sample.labels is None or len(sample.labels) == 0)
            and self.training
        ):
            return None, None  # pragma: no cover

        # load input data
        input_data = self.load_input(
            sample, use_empty=isinstance(sample, FrameGroup)
        )

        if self.training:
            # load annotations to input sample
            self.load_annotation(input_data, sample.labels)

        # apply transforms to input sample
        parameters = self.transform_input(input_data, parameters)

        if not self.training:
            return input_data, parameters

        # postprocess boxes after transforms
        self.postprocess_annotation(
            input_data.images.image_sizes[0], input_data.targets
        )

        if self.cfg.dataloader.skip_empty_samples and input_data.targets.empty:
            return None, None  # pragma: no cover
        return input_data, parameters
