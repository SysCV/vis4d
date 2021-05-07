"""Dataset mapper in openmt."""
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import detectron2.data.detection_utils as d2_utils
import numpy as np
import torch
from detectron2.config import CfgNode
from detectron2.data import transforms as T
from detectron2.data.common import MapDataset as D2MapDataset
from detectron2.data.dataset_mapper import DatasetMapper as D2DatasetMapper
from scalabel.label.typing import Frame, Label

from openmt.common.io import build_data_backend
from openmt.config import DataloaderConfig, ReferenceSamplingConfig
from openmt.struct import Boxes2D, Images, InputSample

from .transforms import build_augmentations
from .utils import dicts_to_boxes2d, im_decode, label_to_dict

__all__ = ["DatasetMapper", "MapDataset"]


class MapDataset(D2MapDataset):  # type: ignore
    """Map a function over the elements in a dataset."""

    def __init__(  # type: ignore
        self, sampling_cfg: ReferenceSamplingConfig, training, *args, **kwargs
    ):
        """Init."""
        super().__init__(*args, **kwargs)
        self.video_to_idcs: Dict[str, List[int]] = defaultdict(list)
        self.frame_to_idcs: Dict[str, Dict[int, int]] = defaultdict(dict)
        self._create_video_mapping()
        self.sampling_cfg = sampling_cfg
        self.training = training

    def _create_video_mapping(self) -> None:
        """Create a mapping that returns all img idx for a given video id."""
        for idx, entry in enumerate(self._dataset):
            if entry["video_name"] is not None:
                self.video_to_idcs[entry["video_name"]].append(idx)

        for video in self.video_to_idcs:
            self.video_to_idcs[video] = sorted(
                self.video_to_idcs[video],
                key=lambda idx: self._dataset[idx]["frame_index"],  # type: ignore # pylint: disable=line-too-long
            )
            self.frame_to_idcs[video] = {
                self._dataset[idx]["frame_index"]: idx
                for idx in self.video_to_idcs[video]
            }

            # assert that frames are a range(0, sequence_length)
            frame_ids = list(self.frame_to_idcs[video].keys())
            assert frame_ids == list(
                range(len(frame_ids))
            ), f"Sequence {video} misses frames: %s" % (
                set(frame_ids) ^ set(list(range(frame_ids[-1] + 1)))
            )

    def sample_ref_idcs(self, video: str, cur_idx: int) -> List[int]:
        """Sample reference indices from video_idcs given cur_idx."""
        frame_ids = list(self.frame_to_idcs[video].keys())
        frame_id = self._dataset[cur_idx]["frame_index"]

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

        return [self.frame_to_idcs[video][f] for f in ref_frame_ids]

    def __getitem__(self, idx: int) -> List[InputSample]:
        """Fully prepare a sample for training/inference."""
        retry_count = 0
        cur_idx = int(idx)

        while True:
            data = self._map_func(self._dataset[cur_idx])
            if data is not None:
                input_data, transforms = data
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
                    return [input_data] + ref_data

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


class DatasetMapper(D2DatasetMapper):  # type: ignore
    """DatasetMapper class for openMT.

    A callable which takes a data sample in scalabel format, and maps it into
    a format used by the openMT model. The callable does the following:
    1. Read image from "url"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations (InputData, AnnotationInstance)
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
        super().__init__(det2cfg, is_train, augmentations=augs)
        self.data_backend = build_data_backend(loader_cfg.data_backend)

    def load_image(
        self,
        sample: Frame,
    ) -> np.ndarray:
        """Load image according to data_backend."""
        assert sample.url is not None
        im_bytes = self.data_backend.get(sample.url)
        image = im_decode(im_bytes)
        sample.size = [image.shape[1], image.shape[0]]
        return image

    def transform_image(
        self,
        image: np.ndarray,
        transforms: Optional[T.AugmentationList] = None,
    ) -> Tuple[Images, T.AugmentationList]:
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
        image_processed = Images(
            torch.as_tensor(
                np.ascontiguousarray(image.transpose(2, 0, 1)),
                dtype=torch.float32,
            ).unsqueeze(0),
            [(image.shape[1], image.shape[0])],
        )
        return image_processed, transforms

    def transform_annotation(
        self,
        input_sample: InputSample,
        labels: Optional[List[Label]],
        transforms: T.AugmentationList,
    ) -> Boxes2D:
        """Transform annotations."""
        image_hw = input_sample.image.tensor.shape[2:]

        if labels is None:
            return Boxes2D(torch.empty(0, 5), torch.empty(0), torch.empty(0))

        # USER: Implement additional transformations if you have other types
        # of data
        annos = []
        for label in labels:
            assert label.attributes is not None
            if not label.attributes.get("ignore", False):
                anno = label_to_dict(label)
                d2_utils.transform_instance_annotations(
                    anno,
                    transforms,
                    image_hw,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                annos.append(anno)

        return dicts_to_boxes2d(annos)

    def __call__(  # type: ignore
        self,
        sample_dict: Dict[str, Any],
        transforms: Optional[T.AugmentationList] = None,
    ) -> Tuple[InputSample, T.AugmentationList]:
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
        image, transforms = self.transform_image(
            self.load_image(sample), transforms=transforms
        )
        input_data = InputSample(sample, image)

        if not self.is_train:  # pragma: no cover
            del sample.labels
            return input_data, transforms

        input_data.instances = self.transform_annotation(
            input_data, sample.labels, transforms
        )
        del sample.labels

        return input_data, transforms
