"""Dataset mapper in vist."""
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import random
import copy
import numpy as np
import torch
from torch.utils import data
from scalabel.label.typing import Frame, ImageSize, Label, Intrinsics as ScalabelIntrinsics, Extrinsics as ScalabelExtrinsics
from scalabel.label.utils import check_crowd, check_ignored, get_matrix_from_extrinsics, get_matrix_from_intrinsics

from vist.common.io import build_data_backend
from ..struct import Boxes2D, Images, InputSample, NDArrayUI8, Intrinsics, Extrinsics, Boxes3D

from .transforms import build_augmentations, AugParams
from .utils import im_decode, prepare_labels, print_class_histogram, discard_labels_outside_set, transform_bbox
from .datasets import BaseDatasetLoader

__all__ = ["ScalabelDataset"]
logger = logging.getLogger(__name__)


class ScalabelDataset(data.Dataset):
    """Preprocess Scalabel format data into VisT input format."""

    def __init__(
        self, dataset: BaseDatasetLoader,
            training: bool, cats_name2id: Optional[Dict[str, int]] = None
    ):
        """Init."""
        self.cfg = dataset.cfg
        self.sampling_cfg = self.cfg.dataloader.ref_sampling
        self.data_backend = build_data_backend(self.cfg.dataloader.data_backend)
        logger.info("Using data backend: %s", self.cfg.dataloader.data_backend.type)

        self.augmentations = build_augmentations(self.cfg.dataloader.transformations)
        logger.info("Augmentations used: %s", self.augmentations)

        for field in self.cfg.dataloader.fields_to_load:
            assert field in ["boxes2d", "boxes3d", "intrinsics", "extrinsics"]

        self._dataset = dataset.frames
        self._dataset_cfg = dataset.metadata_cfg
        self.training = training

        if cats_name2id is not None:
            discard_labels_outside_set(self._dataset, list(cats_name2id.keys()))
        else:
            cats_name2id = {v: i for i, v in enumerate(self._dataset_cfg.categories)}

        frequencies = prepare_labels(
            self._dataset, cats_name2id, self.cfg.dataloader.compute_global_instance_ids
        )
        print_class_histogram(frequencies)

        self._rng = random.Random(42)
        self._fallback_candidates = set(range(len(self._dataset)))

        self.video_to_indices: Dict[str, List[int]] = defaultdict(list)
        self.frame_to_indices: Dict[str, Dict[int, int]] = defaultdict(dict)
        self._create_video_mapping()
        self.training = training

    def __len__(self):
        return len(self._dataset)

    def _create_video_mapping(self) -> None:
        """Create a mapping that returns all img idx for a given video id."""
        for idx, entry in enumerate(self._dataset):
            if entry.video_name is not None:
                self.video_to_indices[entry.video_name].append(idx)

    def sample_ref_idcs(self, video: str, key_dataset_index: int) -> List[int]:
        """Sample reference dataset indices given video and keyframe index."""
        dataset_indices = self.video_to_indices[video]
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
                key=lambda x: x.metadata.frame_index
                if x.metadata.frame_index is not None
                else 0,
            )
        raise NotImplementedError(
            f"Frame ordering {self.sampling_cfg.frame_order} not "
            f"implemented."
        )

    @staticmethod
    def has_matches(
        key_data: InputSample, ref_data: List[InputSample]
    ) -> bool:
        """Check if key / ref data have matches."""
        has_match = False
        assert key_data.boxes2d is not None
        key_track_ids = key_data.boxes2d.track_ids
        for ref_view in ref_data:
            assert isinstance(ref_view.boxes2d, Boxes2D)
            ref_track_ids = ref_view.boxes2d.track_ids
            match = key_track_ids.view(-1, 1) == ref_track_ids.view(1, -1)
            if match.any():
                has_match = True
                break
        return has_match

    def __getitem__(self, idx: int) -> List[InputSample]:
        """Fully prepare a sample for training/inference."""
        retry_count = 0
        cur_idx = int(idx)

        while True:
            data = self.get_sample(self._dataset[cur_idx])
            if data is not None:
                input_data, parameters = data
                if input_data.metadata.attributes is None:
                    input_data.metadata.attributes = dict()
                if self.training:
                    input_data.metadata.attributes["keyframe"] = True
                self._fallback_candidates.add(cur_idx)

                if self.training and self.sampling_cfg.num_ref_imgs > 0:
                    # sample reference views
                    vid_id = input_data.metadata.video_name
                    if vid_id is not None:
                        ref_data = [
                            self.get_sample(
                                self._dataset[ref_idx], parameters=parameters
                            )[0]
                            for ref_idx in self.sample_ref_idcs(
                                vid_id, cur_idx
                            )
                        ]
                    else:
                        ref_data = [  # pragma: no cover
                            input_data
                            for _ in range(self.sampling_cfg.num_ref_imgs)
                        ]

                    if (
                        not self.sampling_cfg.skip_nomatch_samples
                        or self.has_matches(input_data, ref_data)
                    ):
                        return self.sort_samples([input_data] + ref_data)
                else:
                    return [input_data]

            retry_count += 1
            self._fallback_candidates.discard(cur_idx)
            cur_idx = self._rng.sample(self._fallback_candidates, k=1)[0]

            if retry_count >= 5:
                logger.warning(
                    "Failed to get sample for idx: %s, retry count: %s",
                    idx,
                    retry_count,
                )

    def load_image(
        self,
        sample: Frame,
    ) -> NDArrayUI8:
        """Load image according to data_backend."""
        assert sample.url is not None
        im_bytes = self.data_backend.get(sample.url)
        image = im_decode(im_bytes, mode=self.cfg.dataloader.image_channel_mode)
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
            assert len(parameters) == len(self.augmentations), (
                "Length of augmentation parameters must equal the number of "
                "augmentations!"
            )

        transform_matrix = torch.eye(3)
        for i, aug in enumerate(self.augmentations):
            if len(parameters) < len(self.augmentations):
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
            cat_dict = dict()
            for label in labels:
                assert label.attributes is not None
                assert label.category is not None
                if not check_crowd(label) and not check_ignored(label):
                    labels_used.append(label)
                    if label.category not in cat_dict:
                        cat_dict[label.category] = int(
                            label.attributes["category_id"]
                        )

        if "boxes2d" in self.cfg.dataloader.fields_to_load:
            if labels is not None and labels_used:
                boxes2d = Boxes2D.from_scalabel(labels_used, cat_dict)
                boxes2d.boxes[:, :4] = transform_bbox(
                    transform_matrix,
                    boxes2d.boxes[:, :4],
                )
                if self.cfg.dataloader.clip_bboxes_to_image:
                    boxes2d.clip(input_sample.image.image_sizes[0])

                input_sample.boxes2d = boxes2d

        if "boxes3d" in self.cfg.dataloader.fields_to_load:
            if labels is not None and labels_used:
                boxes3d = Boxes3D.from_scalabel(labels_used, cat_dict)
                input_sample.boxes3d = boxes3d

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
    ) -> Optional[Tuple[InputSample, List[AugParams]]]:
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
        input_data = InputSample(copy.deepcopy(sample), image)

        if (
            input_data.metadata.intrinsics is not None
            and "intrinsics" in self.cfg.dataloader.fields_to_load
        ):
            input_data.intrinsics = self.transform_intrinsics(
                input_data.metadata.intrinsics, transform_matrix
            )

        if (
            input_data.metadata.extrinsics is not None
            and "extrinsics" in self.cfg.dataloader.fields_to_load
        ):
            input_data.extrinsics = self.transform_extrinsics(
                input_data.metadata.extrinsics
            )

        if not self.training:
            del input_data.metadata.labels
            return input_data, parameters

        self.transform_annotation(input_data, input_data.metadata.labels, transform_matrix)
        del input_data.metadata.labels

        if (
            self.cfg.dataloader.skip_empty_samples
            and len(input_data.boxes2d) == 0
            and len(input_data.boxes3d) == 0
            and parameters is None
        ):
            return None  # pragma: no cover

        return input_data, parameters
