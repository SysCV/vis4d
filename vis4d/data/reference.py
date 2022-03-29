"""Reference view sampling component of Vis4D datasets."""
import copy
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
from scalabel.label.typing import Frame, FrameGroup

from ..common.registry import RegistryHolder
from ..struct import InputSample
from .mapper import BaseSampleMapper


class BaseReferenceSampler(metaclass=RegistryHolder):
    """Sample reference view indices from dataset given key view index."""

    def __init__(
        self,
        strategy: str = "uniform",
        num_ref_imgs: int = 0,
        scope: int = 1,
        frame_order: str = "key_first",
        skip_nomatch_samples: bool = False,
    ) -> None:
        """Init."""
        self.frames: Optional[List[Frame]] = None
        self.groups: Optional[List[FrameGroup]] = None
        self.strategy = strategy
        self.num_ref_imgs = num_ref_imgs
        self.scope = scope
        self.frame_order = frame_order
        self.skip_nomatch_samples = skip_nomatch_samples

        if scope != 0 and scope < num_ref_imgs // 2:
            raise ValueError("Scope must be higher than num_ref_imgs / 2.")

        if frame_order not in ["key_first", "temporal"]:
            raise ValueError("frame_order must be key_first or temporal.")

        self.video_to_indices: Dict[str, List[int]] = defaultdict(list)
        self.frame_name_to_idx: Dict[str, int] = {}
        self.frame_to_group: Dict[int, int] = {}
        self.frame_to_sensor_id: Dict[int, int] = {}

    def create_mappings(
        self, frames: List[Frame], groups: Optional[List[FrameGroup]] = None
    ) -> None:
        """Creating mappings, e.g. from video id to frame / group indices."""
        video_to_frameidx: Dict[str, List[int]] = defaultdict(list)
        self.frames = frames
        self.groups = groups
        if self.groups is not None:
            for idx, group in enumerate(self.groups):
                if group.videoName is not None:
                    assert (
                        group.frameIndex is not None
                    ), "found videoName but no frameIndex!"
                    video_to_frameidx[group.videoName].append(group.frameIndex)
                    self.video_to_indices[group.videoName].append(idx)
        else:
            for idx, frame in enumerate(self.frames):
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

        if self.groups is not None:
            self.frame_name_to_idx = {
                f.name: i for i, f in enumerate(self.frames)
            }
            for i, g in enumerate(self.groups):
                for sensor_id, fname in enumerate(g.frames):
                    self.frame_to_group[self.frame_name_to_idx[fname]] = i
                    self.frame_to_sensor_id[
                        self.frame_name_to_idx[fname]
                    ] = sensor_id

    def sample_ref_indices(
        self, video: str, key_dataset_index: int
    ) -> List[int]:
        """Sample reference dataset indices given video and keyframe index."""
        dataset_indices = self.video_to_indices[video]
        sensor_id: Optional[int] = None
        if self.groups is not None:
            sensor_id = self.frame_to_sensor_id[key_dataset_index]
            key_dataset_index = self.frame_to_group[key_dataset_index]

        key_index = dataset_indices.index(key_dataset_index)

        if self.strategy == "uniform":
            left = max(0, key_index - self.scope)
            right = min(key_index + self.scope, len(dataset_indices) - 1)
            valid_inds = (
                dataset_indices[left:key_index]
                + dataset_indices[key_index + 1 : right + 1]
            )
            ref_dataset_indices: List[int] = np.random.choice(
                valid_inds, self.num_ref_imgs, replace=False
            ).tolist()
        elif self.strategy == "sequential":
            right = key_index + 1 + self.num_ref_imgs
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
                f"Reference view sampling strategy {self.strategy} not "
                f"implemented."
            )

        if self.groups is not None and sensor_id is not None:
            for i, ref_id in enumerate(ref_dataset_indices):
                fname = self.groups[ref_id].frames[sensor_id]
                ref_dataset_indices[i] = self.frame_name_to_idx[fname]

        return ref_dataset_indices

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

    def __call__(
        self,
        cur_idx: int,
        key_data: InputSample,
        mapper: BaseSampleMapper,
        num_retry: int = 3,
    ) -> Optional[List[InputSample]]:
        """Sample reference views from key view."""
        if self.frames is None:
            raise AttributeError(
                "Please create necessary mappings before using reference "
                "sampler by calling 'create_mappings'!"
            )

        vid_id = key_data.metadata[0].videoName
        for _ in range(num_retry):
            if vid_id is not None and self.scope > 0:
                ref_data = []
                for ref_idx in self.sample_ref_indices(vid_id, cur_idx):
                    ref_sample = mapper(self.frames[ref_idx])
                    if ref_sample is None:
                        break  # pragma: no cover
                    ref_data.append(ref_sample)
            else:  # pragma: no cover
                ref_data = [
                    copy.deepcopy(key_data) for _ in range(self.num_ref_imgs)
                ]
            if (
                not self.skip_nomatch_samples
                or self.has_matches(key_data, ref_data)
            ) and self.num_ref_imgs == len(ref_data):
                return ref_data
        return None
