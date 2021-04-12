"""Dataset mapper for tracking in openmt."""
import copy
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, no_type_check

import detectron2.data.detection_utils as d2_utils
import numpy as np
import torch
from detectron2.config import CfgNode
from detectron2.data import transforms as T
from detectron2.data.common import MapDataset
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.structures import Instances
from pydantic import BaseModel, validator

from openmt.data import utils
from openmt.data.io import DataBackendConfig, build_data_backend

__all__ = ["TrackingDatasetMapper", "MapTrackingDataset"]


class ReferenceSamplingConfig(BaseModel):
    """Config for customizing the sampling for reference views."""

    type: str = "uniform"
    num_ref_imgs: int
    scope: int

    @no_type_check
    @validator("scope")
    def validate_scope(  # pylint: disable=no-self-argument,no-self-use
        cls, value, values
    ):
        """Check scope attribute."""
        if not value > values["num_ref_imgs"] // 2:
            raise ValueError("Scope must be higher than num_ref_imgs / 2.")
        return value


class MapTrackingDataset(MapDataset):  # type: ignore
    """Map a function over the elements in a dataset."""

    def __init__(  # type: ignore
        self, sampling_cfg: ReferenceSamplingConfig, *args, **kwargs
    ):
        """Init."""
        super().__init__(*args, **kwargs)
        self.video_to_idcs: Dict[str, List[int]] = defaultdict(list)
        self._create_video_mapping()
        self.sampling_cfg = sampling_cfg

    def _create_video_mapping(self) -> None:
        """Create a mapping that returns all img idx for a given video id."""
        for idx, entry in enumerate(self._dataset):
            self.video_to_idcs[entry["video_id"]].append(idx)

    def sample_ref_idcs(
        self, video_idcs: List[int], cur_idx: int
    ) -> List[int]:
        """Sample reference indices from video_idcs given cur_idx."""
        frame_to_idx = {self._dataset[i]["frame_id"]: i for i in video_idcs}
        frame_ids = sorted(list(frame_to_idx.keys()))
        frame_id = self._dataset[cur_idx]["frame_id"]

        if self.sampling_cfg.type == "uniform":
            left = max(0, frame_id - self.sampling_cfg.scope)
            right = min(frame_id + self.sampling_cfg.scope, len(frame_ids) - 1)
            valid_inds = (
                frame_ids[left:frame_id] + frame_ids[frame_id + 1 : right + 1]
            )
            ref_frame_ids = np.random.choice(
                valid_inds, self.sampling_cfg.num_ref_imgs, replace=False
            ).tolist()
        elif self.sampling_cfg.type == "sequential":
            right = frame_id + 1 + self.sampling_cfg.num_ref_imgs
            if right <= len(frame_ids):
                ref_frame_ids = frame_ids[frame_id + 1 : right]
            else:
                left = frame_id - (right - len(frame_ids))
                ref_frame_ids = (
                    frame_ids[left:frame_id] + frame_ids[frame_id + 1 :]
                )
        else:
            raise NotImplementedError(
                f"Reference view sampling {self.sampling_cfg.type} not "
                f"implemented."
            )

        return [frame_to_idx[i] for i in ref_frame_ids]

    def __getitem__(self, idx: int) -> List[Dict[str, Any]]:  # type: ignore
        """Fully prepare a sample for training/inference."""
        retry_count = 0
        cur_idx = int(idx)

        while True:
            data = self._map_func(self._dataset[cur_idx])
            if data is not None:
                data_dict, transforms = data
                # sample reference views
                video_idcs = self.video_to_idcs[data_dict["video_id"]]
                ref_data = [
                    self._map_func(
                        self._dataset[ref_idx], transforms=transforms
                    )[0]
                    for ref_idx in self.sample_ref_idcs(video_idcs, cur_idx)
                ]

                self._fallback_candidates.add(cur_idx)
                return [data_dict] + ref_data

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


class TrackingDatasetMapper(DatasetMapper):  # type: ignore
    """DatasetMapper class for tracking.

    A callable which takes a dataset dict in Detectron2 Dataset format,
    and maps it into a format used by the openMT tracking model. The
    callable does the following:
    1. Read image sequence (during train) from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    def __init__(
        self, backend_cfg: DataBackendConfig, det2cfg: CfgNode
    ) -> None:
        """Init."""
        super().__init__(det2cfg)  # pylint: disable=missing-kwoa
        self.data_backend = build_data_backend(backend_cfg)

    def load_image(  # type: ignore
        self, dataset_dict: Dict[str, Any]
    ) -> np.ndarray:
        """Load image according to data_backend."""
        im_bytes = self.data_backend.get(dataset_dict["file_name"])
        image = utils.im_decode(im_bytes)
        d2_utils.check_image_size(dataset_dict, image)
        return image

    def transform_image(  # type: ignore
        self,
        image: np.ndarray,
        dataset_dict: Dict[str, Any],
        transforms: Optional[T.AugmentationList] = None,
    ) -> T.AugmentationList:
        """Apply image augmentations and convert to torch tensor."""
        aug_input = T.AugInput(image)
        if transforms is None:
            transforms = self.augmentations(aug_input)
            image = aug_input.image
        else:
            image = transforms.apply_image(image)

        # Pytorch's dataloader is efficient on torch.Tensor due to
        # shared-memory, but not efficient on large generic data struct due
        # to the use of pickle & mp.Queue. Therefore it's important to use
        # torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        return transforms

    def transform_annotation(  # type: ignore
        self, dataset_dict: Dict[str, Any], transforms: T.AugmentationList
    ) -> Instances:
        """Transform annotations."""
        image_shape = dataset_dict["image"].shape[1:]  # h, w

        # USER: Modify this if you want to keep them for some reason.
        for anno in dataset_dict["annotations"]:
            if not self.use_instance_mask:
                anno.pop("segmentation", None)
            if not self.use_keypoint:
                anno.pop("keypoints", None)

        # USER: Implement additional transformations if you have other types
        # of data
        annos = [
            d2_utils.transform_instance_annotations(
                obj,
                transforms,
                image_shape,
                keypoint_hflip_indices=self.keypoint_hflip_indices,
            )
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        instances = d2_utils.annotations_to_instances(
            annos, image_shape, mask_format=self.instance_mask_format
        )
        instances.set(
            "track_ids",
            torch.tensor(
                [anno["instance_id"] for anno in annos], dtype=torch.int
            ),
        )
        return instances

    def __call__(  # type: ignore
        self, dataset_dict: Dict[str, Any], transforms=None
    ):
        """Prepare a single sample in model format.

        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2
            Dataset format.

        Returns:
            dict: a format that the model accepts
        """
        dataset_dict = copy.deepcopy(
            dataset_dict
        )  # it will be modified by code below

        # image loading
        image = self.load_image(dataset_dict)

        # image augmentation / to torch.tensor
        transforms = self.transform_image(
            image, dataset_dict, transforms=transforms
        )

        if not self.is_train:  # pragma: no cover
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            instances = self.transform_annotation(dataset_dict, transforms)
            dataset_dict["instances"] = d2_utils.filter_empty_instances(
                instances
            )

        return dataset_dict, transforms
