"""DatasetHandler that manages one / multiple dataset(s)."""
from collections import defaultdict
from typing import Callable, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils import data

from vis4d.common.registry import RegistryHolder, build_component
from vis4d.data.transforms import BaseAugmentation
from vis4d.struct import InputSample, LabelInstances, ModuleCfg


def sort_by_frame_index(samples: List[InputSample]) -> List[InputSample]:
    """Sort samples by frameIndex (i.e. temporal order), if applicable."""
    return sorted(
        samples,
        key=lambda x: x.metadata[0].frameIndex
        if x.metadata[0].frameIndex is not None
        else 0,
    )


class Vis4DDatasetHandler(data.ConcatDataset, metaclass=RegistryHolder):  # type: ignore # pylint: disable=line-too-long
    """DatasetHandler class.

    This class wraps one or multiple instances of ScalabelDataset so that the
    augmentation and annotation postprocessing settings can be shared across
    those datasets.
    """

    def __init__(
        self,
        datasets: Iterable[data.Dataset],
        clip_bboxes_to_image: bool = True,
        min_bboxes_area: float = 7.0 * 7.0,
        transformations: Optional[
            List[Union[BaseAugmentation, ModuleCfg]]
        ] = None,
        sample_sort: Callable[
            [List[InputSample]], List[InputSample]
        ] = lambda x: x,
    ) -> None:
        """Init."""
        super().__init__(datasets)
        self.clip_bboxes_to_image = clip_bboxes_to_image
        self.min_bboxes_area = min_bboxes_area
        self.sample_sort = sample_sort
        self.transformations = []
        # TODO Temporary fix to separate augmentation btw group and frames, will be removed once split the dataset into single vs multi sensor # pylint: disable=line-too-long,fixme
        if len(datasets) == 1:  # type: ignore
            self.use_group = (
                not datasets[0].mapper.training  # type: ignore
                and datasets[0].dataset.groups is not None  # type: ignore
            )
        else:
            self.use_group = False
        if transformations is not None:
            for transform in transformations:
                if isinstance(transform, dict):
                    transform_: BaseAugmentation = build_component(
                        transform, bound=BaseAugmentation
                    )
                else:  # pragma: no cover
                    transform_ = transform
                self.transformations.append(transform_)

    def _postprocess_annotations(
        self, im_wh: Tuple[int, int], targets: LabelInstances
    ) -> None:
        """Process annotations after transform."""
        if len(targets.boxes2d[0]) == 0:
            return
        if self.clip_bboxes_to_image:
            targets.boxes2d[0].clip(im_wh)
        keep = targets.boxes2d[0].area >= self.min_bboxes_area
        targets.boxes2d = [targets.boxes2d[0][keep]]
        if len(targets.boxes3d[0]) > 0:
            targets.boxes3d = [targets.boxes3d[0][keep]]
        if len(targets.instance_masks[0]) > 0:
            targets.instance_masks = [targets.instance_masks[0][keep]]

    @staticmethod
    def _rescale_track_ids(samples: List[InputSample]) -> None:
        """If multiple samples were combined, rescale track ids.

        The track ids will be rescaled to a contiguous array starting at 0
        """
        # gather all track ids
        track_ids_all = defaultdict(list)
        for s in samples:
            for l_list in s.targets.get_instance_labels():
                for l in l_list:  # type: ignore
                    if hasattr(l, "track_ids"):
                        track_ids_all[type(l)].append(l.track_ids)

        # rescale
        for key, track_list in track_ids_all.items():
            all_tracks = list(torch.cat(track_list))
            new_track_list = []
            for track in track_list:
                new_track = []
                for idx in track:
                    new_track.append(all_tracks.index(idx))
                new_track_list.append(torch.tensor(new_track))
            track_ids_all[key] = new_track_list

        # replace
        for i, s in enumerate(samples):
            for l_list in s.targets.get_instance_labels():
                for l in l_list:  # type: ignore
                    if hasattr(l, "track_ids"):
                        l.track_ids = track_ids_all[type(l)][i]

    def __getitem__(self, idx: int) -> List[InputSample]:
        """Wrap getitem to apply augmentations."""
        getitem = super().__getitem__
        samples = getitem(idx)
        for aug in self.transformations:
            if aug.num_samples > 1:
                idcs = np.random.randint(0, len(self), aug.num_samples - 1)
                addsamples = [getitem(i) for i in idcs]
            else:
                addsamples = None

            params = None
            for samp_i, sample in enumerate(samples):
                if addsamples is not None:
                    sample = InputSample.cat(
                        [sample, *[s[samp_i] for s in addsamples]]
                    )
                # TODO Temporary fix to separate augmentation btw group and frames, will be removed once split the dataset into single vs multi sensor # pylint: disable=line-too-long,fixme
                if params is None and not (self.use_group and samp_i == 0):
                    params = aug.generate_parameters(sample)
                samples[samp_i], _ = aug(sample, params)

            if addsamples is not None:
                self._rescale_track_ids(samples)

        for s in samples:
            self._postprocess_annotations(s.images.image_sizes[0], s.targets)

        return self.sample_sort(samples)
