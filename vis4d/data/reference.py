"""Reference view sampling component of Vis4D datasets."""
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
from pydantic import BaseModel, validator
from scalabel.label.typing import Frame, FrameGroup

from ..common.registry import RegistryHolder
from ..struct import InputSample
from .mapper import BaseSampleMapper
from .transforms import AugParams


class ReferenceSamplerConfig(BaseModel):
    """Config for customizing the sampling of reference views."""

    type: str = "BaseReferenceSampler"
    strategy: str = "uniform"
    num_ref_imgs: int = 0
    scope: int = 1
    frame_order: str = "key_first"
    skip_nomatch_samples: bool = False

    @validator("scope")
    def validate_scope(  # type: ignore # pylint: disable=no-self-argument,no-self-use, line-too-long
        cls, value: int, values
    ) -> int:
        """Check scope attribute."""
        if value != 0 and value < values["num_ref_imgs"] // 2:
            raise ValueError("Scope must be higher than num_ref_imgs / 2.")
        return value

    @validator("frame_order")
    def validate_frame_order(  # pylint: disable=no-self-argument,no-self-use
        cls, value: str
    ) -> str:
        """Check frame_order attribute."""
        if not value in ["key_first", "temporal"]:
            raise ValueError("frame_order must be key_first or temporal.")
        return value


class BaseReferenceSampler(metaclass=RegistryHolder):
    """Sample reference view indices from dataset given key view index."""

    def __init__(
        self,
        cfg: ReferenceSamplerConfig,
        frames: List[Frame],
        groups: Optional[List[FrameGroup]] = None,
    ) -> None:
        """Init."""
        self.cfg = cfg
        self.frames = frames
        self.groups = groups
        self.video_to_indices: Dict[str, List[int]] = defaultdict(list)
        self._create_video_mapping()

        if self.groups is not None:
            self.frame_name_to_idx = {
                f.name: i for i, f in enumerate(self.frames)
            }
            self.frame_to_group: Dict[int, int] = {}
            self.frame_to_sensor_id: Dict[int, int] = {}
            for i, g in enumerate(self.groups):
                for sensor_id, fname in enumerate(g.frames):
                    self.frame_to_group[self.frame_name_to_idx[fname]] = i
                    self.frame_to_sensor_id[
                        self.frame_name_to_idx[fname]
                    ] = sensor_id

    def _create_video_mapping(self) -> None:
        """Creating mapping from video id to frame / group indices."""
        video_to_frameidx: Dict[str, List[int]] = defaultdict(list)
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

    def sort_samples(
        self, input_samples: List[InputSample]
    ) -> List[InputSample]:
        """Sort samples according to sampling cfg."""
        if self.cfg.frame_order == "key_first":
            return input_samples
        if self.cfg.frame_order == "temporal":
            return sorted(
                input_samples,
                key=lambda x: x.metadata[0].frameIndex
                if x.metadata[0].frameIndex is not None
                else 0,
            )
        raise NotImplementedError(
            f"Frame ordering {self.cfg.frame_order} not " f"implemented."
        )

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

        if self.cfg.strategy == "uniform":
            left = max(0, key_index - self.cfg.scope)
            right = min(key_index + self.cfg.scope, len(dataset_indices) - 1)
            valid_inds = (
                dataset_indices[left:key_index]
                + dataset_indices[key_index + 1 : right + 1]
            )
            ref_dataset_indices: List[int] = np.random.choice(
                valid_inds, self.cfg.num_ref_imgs, replace=False
            ).tolist()
        elif self.cfg.strategy == "sequential":
            right = key_index + 1 + self.cfg.num_ref_imgs
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
                f"Reference view sampling strategy {self.cfg.strategy} not "
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
        parameters: Optional[List[AugParams]],
        num_retry: int = 3,
    ) -> Optional[List[InputSample]]:
        """Sample reference views from key view."""
        vid_id = key_data.metadata[0].videoName
        for _ in range(num_retry):
            if vid_id is not None:
                ref_data = []
                for ref_idx in self.sample_ref_indices(vid_id, cur_idx):
                    ref_sample = mapper(
                        self.frames[ref_idx],
                        parameters=parameters,
                    )[0]
                    if ref_sample is None:
                        break  # pragma: no cover
                    ref_data.append(ref_sample)
            else:  # pragma: no cover
                if parameters is not None:
                    ref_data = [key_data for _ in range(self.cfg.num_ref_imgs)]
                else:
                    ref_data = []
                    for _ in range(self.cfg.num_ref_imgs):
                        ref_sample = mapper(self.frames[cur_idx])[0]
                        if ref_sample is None:
                            break
                        ref_data.append(ref_sample)
            if (
                not self.cfg.skip_nomatch_samples
                or self.has_matches(key_data, ref_data)
            ) and self.cfg.num_ref_imgs == len(ref_data):
                return ref_data
        return None


def build_reference_sampler(
    cfg: ReferenceSamplerConfig,
    frames: List[Frame],
    groups: Optional[List[FrameGroup]] = None,
) -> BaseReferenceSampler:
    """Build a reference view sampler."""
    registry = RegistryHolder.get_registry(BaseReferenceSampler)
    if cfg.type in registry:
        module = registry[cfg.type](cfg, frames, groups)
        assert isinstance(module, BaseReferenceSampler)
        return module
    raise NotImplementedError(f"Reference sampler type {cfg.type} not found.")
