"""Build Vis4D data loading pipeline."""
from typing import List, Optional, Union, Iterable, Tuple

import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_info
import torch
import numpy as np
from torch.utils import data
from torch.utils.data.distributed import DistributedSampler

from ..common.registry import RegistryHolder, build_component
from ..common.utils import get_world_size
from .transforms import BaseAugmentation
from .dataset import ScalabelDataset
from .samplers import BaseSampler, TrackingInferenceSampler
from .utils import identity_batch_collator
from ..struct import ModuleCfg, LabelInstances, InputSample


class Vis4DDatasetHandler(data.ConcatDataset):
    def __init__(
        self,
        datasets: Iterable[data.Dataset],
        clip_bboxes_to_image: bool = True,
        min_bboxes_area: float = 7.0 * 7.0,
        transformations: Optional[
            List[Union[BaseAugmentation, ModuleCfg]]
        ] = None,
    ) -> None:
        """Init."""
        super().__init__(datasets)
        self.clip_bboxes_to_image = clip_bboxes_to_image
        self.min_bboxes_area = min_bboxes_area
        self.transformations = []
        if transformations is not None:
            for transform in transformations:
                if isinstance(transform, dict):
                    transform_: BaseAugmentation = build_component(
                        transform, bound=BaseAugmentation
                    )
                else:  # pragma: no cover
                    transform_ = transform
                self.transformations.append(transform_)
        rank_zero_info("Transformations used: %s", self.transformations)

    def postprocess_annotations(
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

    def sort_samples(self, samples: List[InputSample]) -> List[InputSample]:
        """Sort samples according to sampling cfg."""
        if self.frame_order == "key_first":
            return samples
        if self.frame_order == "temporal":
            return sorted(
                samples,
                key=lambda x: x.metadata[0].frameIndex
                if x.metadata[0].frameIndex is not None
                else 0,
            )
        raise NotImplementedError(
            f"Frame ordering {self.frame_order} not implemented."
        )

    def __getitem__(self, idx):
        """Wrap getitem to apply augmentations."""
        getitem = super().__getitem__
        samples = getitem(idx)
        for aug_i, aug in enumerate(self.transformations):
            if aug.num_samples > 1:
                idcs = np.random.randint(0, len(self), aug.num_samples - 1)
                addsamples = [getitem(i) for i in idcs]
            else:
                addsamples = None

            params = None
            for samp_i, sample in enumerate(samples):
                if addsamples is not None:
                    sample = InputSample.cat([sample, *[s[samp_i] for s in addsamples]])
                if params is None:
                    params = aug.generate_parameters(sample)
                samples[samp_i], _ = aug(sample, params)

        for s in samples:
            self.postprocess_annotations(s.images.image_sizes[0], s.targets)
        # TODO sort samples integration
        return samples


class Vis4DDataModule(pl.LightningDataModule, metaclass=RegistryHolder):
    """Default Data module for Vis4D."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        samples_per_gpu: int,
        workers_per_gpu: int,
        train_datasets: Optional[Vis4DDatasetHandler] = None,
        test_datasets: Optional[List[Vis4DDatasetHandler]] = None,
        predict_datasets: Optional[List[Vis4DDatasetHandler]] = None,
        seed: Optional[int] = None,
        pin_memory: bool = False,
        train_sampler: Optional[BaseSampler] = None,
    ) -> None:
        """Init."""
        super().__init__()  # type: ignore
        self.samples_per_gpu = samples_per_gpu
        self.workers_per_gpu = workers_per_gpu
        self.seed = seed
        self.pin_memory = pin_memory
        self.train_datasets = train_datasets
        self.test_datasets = test_datasets
        self.predict_datasets = predict_datasets
        self.train_sampler = train_sampler

    def train_dataloader(self) -> data.DataLoader:
        """Return dataloader for training."""
        assert self.train_datasets is not None, "No train datasets specified!"
        # TODO option to customize DatasetHandler
        if self.train_sampler is not None:
            batch_size, shuffle = 1, False
        else:
            batch_size, shuffle = self.samples_per_gpu, True
        train_dataloader = data.DataLoader(
            self.train_datasets,
            batch_sampler=self.train_sampler,
            batch_size=batch_size,
            num_workers=self.workers_per_gpu,
            collate_fn=identity_batch_collator,
            persistent_workers=self.workers_per_gpu > 0,
            pin_memory=self.pin_memory,
            shuffle=shuffle,
        )
        return train_dataloader

    def predict_dataloader(
        self,
    ) -> Union[data.DataLoader, List[data.DataLoader]]:
        """Return dataloader(s) for prediction."""
        if self.predict_datasets is not None:
            return self._build_inference_dataloaders(self.predict_datasets)
        return self.test_dataloader()  # pragma: no cover

    def val_dataloader(self) -> List[data.DataLoader]:
        """Return dataloaders for validation."""
        return self.test_dataloader()

    def test_dataloader(self) -> List[data.DataLoader]:
        """Return dataloaders for testing."""
        assert self.test_datasets is not None, "No test datasets specified!"
        return self._build_inference_dataloaders(self.test_datasets)

    def transfer_batch_to_device(
        self,
        batch: List[List[InputSample]],
        device: torch.device,
        dataloader_idx: int,
    ) -> List[InputSample]:
        """Put input in correct format for model, move to device."""
        # group by ref views by sequence: NxM --> MxN, where M=num_refs, N=BS
        batch = [
            [batch[j][i] for j in range(len(batch))]
            for i in range(len(batch[0]))
        ]
        return [InputSample.cat(elem, device) for elem in batch]

    def _build_inference_dataloaders(
        self, datasets: List[ScalabelDataset]
    ) -> List[data.DataLoader]:
        """Build dataloaders for test / predict."""
        dataloaders = []
        for dataset in datasets:
            sampler: Optional[data.Sampler] = None
            if get_world_size() > 1:# and dataset.has_sequences: TODO fix
                sampler = TrackingInferenceSampler(dataset)  # pragma: no cover
            elif get_world_size() > 1 and self.train_sampler is not None:
                # manually create distributed sampler for inference if using
                # custom training sampler
                sampler = DistributedSampler(  # pragma: no cover
                    dataset, shuffle=False
                )

            test_dataloader = data.DataLoader(
                dataset,
                batch_size=1,
                num_workers=self.workers_per_gpu,
                sampler=sampler,
                collate_fn=identity_batch_collator,
                persistent_workers=self.workers_per_gpu > 0,
            )
            dataloaders.append(test_dataloader)
        return dataloaders
