"""Dataset mapper in vist."""
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from detectron2.config import CfgNode
from detectron2.data import transforms as T
from detectron2.data.common import MapDataset as D2MapDataset
from detectron2.data.dataset_mapper import DatasetMapper as D2DatasetMapper
from kornia.augmentation import AugmentationBase2D
from kornia.geometry.bbox import transform_bbox
from pydantic import validator
from pydantic.main import BaseModel
from scalabel.label.typing import Frame, ImageSize, Intrinsics, Label
from scalabel.label.utils import (
    check_crowd,
    check_ignored,
    get_matrix_from_intrinsics,
)

from vist.common.io import build_data_backend
from vist.struct import Boxes2D, Images, InputSample, NDArrayUI8

from ..common.io import DataBackendConfig
from .transforms import AugmentationConfig, build_augmentations
from .utils import im_decode

__all__ = ["DatasetMapper", "MapDataset"]


class ReferenceSamplingConfig(BaseModel):
    """Config for customizing the sampling of reference views."""

    type: str = "uniform"
    num_ref_imgs: int = 0
    scope: int = 1
    frame_order: str = "key_first"
    skip_nomatch_samples: bool = False

    @validator("scope")
    def validate_scope(  # type: ignore # pylint: disable=no-self-argument,no-self-use, line-too-long
        cls, value: int, values
    ) -> int:
        """Check scope attribute."""
        if not value > values["num_ref_imgs"] // 2:
            raise ValueError("Scope must be higher than num_ref_imgs / 2.")
        return value


class MapDataset(D2MapDataset):  # type: ignore
    """Map a function over the elements in a dataset."""

    def __init__(  # type: ignore
        self, sampling_cfg: ReferenceSamplingConfig, training, *args, **kwargs
    ):
        """Init."""
        super().__init__(*args, **kwargs)
        self.video_to_indices: Dict[str, List[int]] = defaultdict(list)
        self.frame_to_indices: Dict[str, Dict[int, int]] = defaultdict(dict)
        self._create_video_mapping()
        self.sampling_cfg = sampling_cfg
        self.training = training

    def _create_video_mapping(self) -> None:
        """Create a mapping that returns all img idx for a given video id."""
        for idx, entry in enumerate(self._dataset):
            if entry["video_name"] is not None:
                self.video_to_indices[entry["video_name"]].append(idx)

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
            data = self._map_func(self._dataset[cur_idx])
            if data is not None:
                input_data, transforms = data
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
                            self._map_func(
                                self._dataset[ref_idx], transforms=transforms
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

            # _map_func fails for this idx, use a random new index from the
            # pool
            retry_count += 1
            self._fallback_candidates.discard(cur_idx)
            cur_idx = self._rng.sample(self._fallback_candidates, k=1)[0]

            if retry_count >= 5:
                logger = logging.getLogger(__name__)
                logger.warning(
                    "Failed to apply `_map_func` for idx: %s, retry count: %s",
                    idx,
                    retry_count,
                )


class DataloaderConfig(BaseModel):
    """Config for dataloader."""

    data_backend: DataBackendConfig = DataBackendConfig()
    workers_per_gpu: int
    categories: Optional[List[str]] = None
    skip_empty_samples: bool = False
    clip_bboxes_to_image: bool = True
    compute_global_instance_ids: bool = False
    train_augmentations: Optional[List[AugmentationConfig]] = None
    test_augmentations: Optional[List[AugmentationConfig]] = None
    ref_sampling_cfg: ReferenceSamplingConfig
    image_channel_mode: str
    ignore_unknown_cats: bool = False


class DatasetMapper(D2DatasetMapper):  # type: ignore
    """DatasetMapper class for VisT.

    A callable which takes a data sample in scalabel format, and maps it into
    a format used by the VisT model. The callable does the following:
    1. Read image from "url"
    2. Applies transforms to the image and annotations
    3. Put data and annotations in VisT format (InputSample)
    """

    def __init__(
        self,
        loader_cfg: DataloaderConfig,
        det2cfg: CfgNode,
        is_train: bool = True,
    ) -> None:
        """Init."""
        # pylint: disable=missing-kwoa,too-many-function-args
        if is_train:
            augs = build_augmentations(loader_cfg.train_augmentations)
        else:
            augs = build_augmentations(loader_cfg.test_augmentations)
        super().__init__(det2cfg, is_train)
        self.augs = augs
        self.loader_cfg = loader_cfg
        self.data_backend = build_data_backend(loader_cfg.data_backend)
        self.skip_empty_samples = loader_cfg.skip_empty_samples

    def load_image(
        self,
        sample: Frame,
    ) -> NDArrayUI8:
        """Load image according to data_backend."""
        assert sample.url is not None
        im_bytes = self.data_backend.get(sample.url)
        image = im_decode(im_bytes, mode=self.loader_cfg.image_channel_mode)
        sample.size = ImageSize(width=image.shape[1], height=image.shape[0])
        return image

    def transform_image(
        self,
        image: NDArrayUI8,
        transforms: List[AugmentationBase2D] = None,
    ):
        """Apply image augmentations and convert to torch tensor."""
        transform_matrix = torch.eye(3, 3)

        image = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1)),
            dtype=torch.float32,
        ).unsqueeze(0)

        if transforms is None:
            transforms = self.augs

        for aug in transforms:
            image, tm = aug(image)
            transform_matrix = torch.mm(tm[0], transform_matrix)

        image_processed = Images(image, [(image.shape[3], image.shape[2])])

        return image_processed, transforms, transform_matrix

    def transform_annotation(
        self,
        input_sample: InputSample,
        labels: Optional[List[Label]],
        transform_matrix: np.ndarray,
    ) -> Boxes2D:
        """Transform annotations."""
        image_hw = input_sample.image.tensor.shape[2:]

        labels_used = []

        if labels is not None:
            cat_dict = dict()
            for label in labels:
                assert label.attributes is not None
                if not check_crowd(label) and not check_ignored(label):
                    labels_used.append(label)
                    if label.category not in cat_dict:
                        cat_dict[label.category] = label.attributes[
                            "category_id"
                        ]

            if labels_used:
                boxes2d = Boxes2D.from_scalabel(labels_used, cat_dict)
                # from openmt.struct import Boxes3D
                # boxes3d = Boxes3D.from_scalabel(labels_used, cat_dict)

        if labels is None or not labels_used:
            return Boxes2D(torch.empty(0, 5), torch.empty(0), torch.empty(0))

        bboxes_2d = boxes2d.boxes[:, :4]

        for i in range(bboxes_2d.shape[0]):
            bboxes_2d[i] = transform_bbox(
                transform_matrix.unsqueeze(0), bboxes_2d[i].unsqueeze(0)
            )

            # clip transformed bbox to image size
            if self.loader_cfg.clip_bboxes_to_image:
                bboxes_2d[i].clip(min=0)
                bboxes_2d[i] = np.minimum(
                    bboxes_2d[i], list(image_hw + image_hw)[::-1]
                )

        boxes2d.boxes[:, :4] = bboxes_2d

        return boxes2d

    def __call__(  # type: ignore
        self,
        sample_dict: Dict[str, Any],
        transforms: Optional[T.AugmentationList] = None,
    ) -> Optional[Tuple[InputSample, T.AugmentationList]]:
        """Prepare a single sample in detect format.

        Args:
            sample_dict (serialized Frame): Metadata of one image, in scalabel
            format. Serialized as dict due to multi-processing.
            transforms (T.AugmentationList): Detectron2 augmentation list.

        Returns:
            InputSample: Data format that the model accepts.
            T.AugmentationList: augmentations, s.t. ref views can be augmented
            with the same parameters.
        """
        sample = Frame(**sample_dict)

        # image loading, augmentation / to torch.tensor
        image, transforms, transform_matrix = self.transform_image(
            self.load_image(sample),
            transforms=transforms,
        )
        input_data = InputSample(sample, image)

        if not self.is_train:
            del sample.labels
            return input_data, transforms

        intrinsic_matrix = torch.from_numpy(
            get_matrix_from_intrinsics(sample.intrinsics)
        ).to(torch.float32)

        projection = torch.mm(transform_matrix, intrinsic_matrix).numpy()

        input_data.metadata.intrinsics = Intrinsics(
            focal=(projection[0][0], projection[1][1]),
            center=(projection[0][2], projection[1][2]),
        )

        input_data.boxes2d = self.transform_annotation(
            input_data, sample.labels, transform_matrix
        )
        del sample.labels

        if (
            self.skip_empty_samples
            and len(input_data.boxes2d) == 0
            and transforms is None
        ):
            return None  # pragma: no cover

        return input_data, transforms
