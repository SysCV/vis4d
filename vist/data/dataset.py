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
from scalabel.label.typing import Frame, ImageSize
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

from vist.common.io import build_data_backend

from ..struct import (
    Boxes2D,
    Boxes3D,
    Extrinsics,
    Images,
    InputSample,
    Intrinsics,
    NDArrayUI8,
)
from .datasets import BaseDatasetLoader
from .transforms import AugParams, build_augmentations
from .utils import (
    discard_labels_outside_set,
    im_decode,
    prepare_labels,
    print_class_histogram,
    transform_bbox,
    filter_attributes
)

__all__ = ["ScalabelDataset"]


class ScalabelDataset(Dataset):  # type: ignore
    """Preprocess Scalabel format data into VisT input format."""

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

        for field in self.cfg.dataloader.fields_to_load:
            assert field in ["boxes2d", "boxes3d", "intrinsics", "extrinsics"]

        self.dataset = dataset
        self.training = training

        if cats_name2id is not None:
            discard_labels_outside_set(
                self.dataset.frames, list(cats_name2id.keys())
            )
        else:
            cats_name2id = {
                v: i
                for i, v in enumerate(
                    [
                        c.name
                        for c in get_leaf_categories(
                            self.dataset.metadata_cfg.categories
                        )
                    ]
                )
            }
        self.cats_name2id = cats_name2id
        self.dataset.frames = filter_attributes(self.dataset.frames, self.dataset.cfg.attributes)

        frequencies = prepare_labels(
            self.dataset.frames,
            cats_name2id,
            self.cfg.dataloader.compute_global_instance_ids,
        )
        print_class_histogram(frequencies)

        self.has_groups = dataset.groups is not None
        self._fallback_candidates = set(range(len(self.dataset.frames)))
        self.video_to_indices: Dict[str, List[int]] = defaultdict(list)
        self._create_video_mapping()
        self.has_sequences = bool(self.video_to_indices)

        if self.has_groups:
            self.frame_name_to_idx = {f.name: i for i, f in enumerate(self.dataset.frames)}
            self.frame_to_group: Dict[int, int] = {}
            self.frame_to_sensor_id: Dict[int, int] = {}
            for i, g in enumerate(self.dataset.groups):
                for sensor_id, fname in enumerate(g.frames):
                    self.frame_to_group[self.frame_name_to_idx[fname]] = i
                    self.frame_to_sensor_id[self.frame_name_to_idx[fname]] = sensor_id

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.dataset.frames)

    def _create_video_mapping(self) -> None:
        """Creating mapping from video id to frame / group indices."""
        # TODO sort by frame id
        if self.has_groups:
            for idx, entry in enumerate(self.dataset.groups):
                if entry.videoName is not None:
                    self.video_to_indices[entry.videoName].append(idx)
        else:
            for idx, entry in enumerate(self.dataset.frames):
                if entry.videoName is not None:
                    self.video_to_indices[entry.videoName].append(idx)

    def sample_ref_indices(
        self, video: str, key_dataset_index: int
    ) -> List[int]:
        """Sample reference dataset indices given video and keyframe index."""
        dataset_indices = self.video_to_indices[video]
        sensor_id: Optional[int] = None
        if self.has_groups:
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
            ref_dataset_indices = np.random.choice(
                valid_inds, self.sampling_cfg.num_ref_imgs, replace=False
            ).tolist()  # type: List[int]
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

        if sensor_id is not None:
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
    ) -> Optional[List[InputSample]]:
        """Sample reference views from key view."""
        vid_id = key_data.metadata[0].videoName
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
        else:
            ref_data = [  # pragma: no cover
                key_data for _ in range(self.sampling_cfg.num_ref_imgs)
            ]

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
        key_track_ids = key_data.boxes2d[0].track_ids
        for ref_view in ref_data:
            ref_track_ids = ref_view.boxes2d[0].track_ids
            match = key_track_ids.view(-1, 1) == ref_track_ids.view(1, -1)
            if match.any():
                return True
        return False  # pragma: no cover

    def __getitem__(self, idx: int) -> List[InputSample]:
        """Fully prepare a sample for training/inference."""
        retry_count = 0
        cur_idx = int(idx)

        if not self.training:
            if self.has_groups:
                group = self.dataset.groups[self.frame_to_group[idx]]
                input_data = []
                for fname in group.frames:
                    cur_data, _ = self.get_sample(
                        self.dataset.frames[self.frame_name_to_idx[fname]]
                    )
                    input_data.append(cur_data)
                return input_data
            else:
                input_data, _ = self.get_sample(
                    self.dataset.frames[cur_idx]
                )
                return [input_data]
        else:
            while True:
                input_data, parameters = self.get_sample(
                    self.dataset.frames[cur_idx]
                )
                if input_data is not None:
                    if input_data.metadata[0].attributes is None:
                        input_data.metadata[0].attributes = {}
                    if self.training:
                        input_data.metadata[0].attributes["keyframe"] = True

                    if self.training and self.sampling_cfg.num_ref_imgs > 0:
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
                        f"Failed to get sample for idx: {idx}, "
                        f"retry count: {retry_count}"
                    )

    def load_image(
        self,
        sample: Frame,
    ) -> NDArrayUI8:
        """Load image according to data_backend."""
        assert sample.url is not None
        im_bytes = self.data_backend.get(sample.url)
        image = im_decode(im_bytes, mode=self.image_channel_mode)
        sample.size = ImageSize(width=image.shape[1], height=image.shape[0])
        return image

    def transform_image(
        self,
        image: NDArrayUI8,
        parameters: Optional[List[AugParams]] = None,
    ) -> Tuple[Images, List[AugParams], torch.Tensor]:
        """Apply image augmentations and convert to torch tensor."""
        image = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1)),
            dtype=torch.float32,
        ).unsqueeze(0)

        if parameters is None:
            parameters = []
        else:
            assert len(parameters) == len(self.transformations), (
                "Length of augmentation parameters must equal the number of "
                "augmentations!"
            )

        transform_matrix = torch.eye(3)
        for i, aug in enumerate(self.transformations):
            if len(parameters) < len(self.transformations):
                parameters.append(aug.forward_parameters(image.shape))
            image, tm = aug(image, parameters[i], return_transform=True)
            transform_matrix = torch.mm(tm[0], transform_matrix)

        image_processed = Images(image, [(image.shape[3], image.shape[2])])
        return image_processed, parameters, transform_matrix

    def transform_annotation(
        self,
        input_sample: InputSample,
        labels: Optional[List[Label]],
        transform_matrix: torch.Tensor,
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

            if "boxes2d" in self.cfg.dataloader.fields_to_load and labels_used:
                boxes2d = Boxes2D.from_scalabel(
                    labels_used, category_dict, instance_id_dict
                )
                boxes2d.boxes[:, :4] = transform_bbox(
                    transform_matrix,
                    boxes2d.boxes[:, :4],
                )
                if self.cfg.dataloader.clip_bboxes_to_image:
                    boxes2d.clip(input_sample.images.image_sizes[0])

                input_sample.boxes2d = [boxes2d]

            if "boxes3d" in self.cfg.dataloader.fields_to_load and labels_used:
                boxes3d = Boxes3D.from_scalabel(
                    labels_used, category_dict, instance_id_dict
                )
                input_sample.boxes3d = [boxes3d]

    @staticmethod
    def transform_intrinsics(
        intrinsics: ScalabelIntrinsics, transform_matrix: torch.Tensor
    ) -> Intrinsics:
        """Transform intrinsic camera matrix according to augmentations."""
        intrinsic_matrix = torch.from_numpy(
            get_matrix_from_intrinsics(intrinsics)
        ).to(torch.float32)
        return Intrinsics(torch.mm(transform_matrix, intrinsic_matrix))

    @staticmethod
    def transform_extrinsics(extrinsics: ScalabelExtrinsics) -> Extrinsics:
        """Transform extrinsics from Scalabel to VisT."""
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
        # image loading, augmentation / to torch.tensor
        image, parameters, transform_matrix = self.transform_image(
            self.load_image(sample),
            parameters=parameters,
        )
        input_data = InputSample([copy.deepcopy(sample)], image)

        if (
            sample.intrinsics is not None
            and "intrinsics" in self.cfg.dataloader.fields_to_load
        ):
            input_data.intrinsics = self.transform_intrinsics(
                sample.intrinsics, transform_matrix
            )

        if (
            sample.extrinsics is not None
            and "extrinsics" in self.cfg.dataloader.fields_to_load
        ):
            input_data.extrinsics = self.transform_extrinsics(
                sample.extrinsics
            )

        if not self.training:
            return input_data, parameters

        self.transform_annotation(input_data, sample.labels, transform_matrix)

        if (
            self.cfg.dataloader.skip_empty_samples
            and len(input_data.boxes2d[0]) == 0
            and len(input_data.boxes3d[0]) == 0
        ):
            return None, None  # pragma: no cover

        return input_data, parameters
