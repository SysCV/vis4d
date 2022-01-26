"""Build Vis4D data loading pipeline."""
from typing import List, Optional, Union

import pytorch_lightning as pl
import torch
from torch.utils import data
from torch.utils.data.distributed import DistributedSampler

from ..common.registry import RegistryHolder
from ..common.utils import get_world_size
from ..struct import InputSample
from .dataset import ScalabelDataset
from .samplers import BaseSampler, TrackingInferenceSampler
from .utils import identity_batch_collator


class Vis4DDatasetHandler(data.ConcatDataset):

    def __init__(self, datasets: Iterable[Dataset],
        clip_bboxes_to_image: bool = True,
        min_bboxes_area: float = 7.0 * 7.0,
         transformations: Optional[List[Union[BaseAugmentation, ModuleCfg]]] = None
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

    def transform_inputs(
            self,
            samples: List[InputSample],
    ) -> None:
        """Apply transforms to input samples.

        Args:
            samples: Input sample and (possibly) reference views.
        """
        parameters = []
        for sample in samples:
            for i, aug in enumerate(self.transformations):
                if len(parameters) < len(self.transformations):
                    if aug.num_samples > 1:
                        idcs = np.random.randint(0, len(self), aug.num_samples)
                        addsamples = [super().__getitem__(idx) for idx in idcs]
                        sample = InputSample.cat([sample, *addsamples])
                    sample, params = aug(sample)
                    parameters.append(params)
                else:
                    sample, _ = aug(sample, parameters[i])

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

    def augment_data(self, data: List[InputSample], training: bool) -> List[InputSample]:
        # apply transforms to input sample
        parameters = self.transform_inputs(input_data, parameters)

        if not training:
            return input_data, parameters

        # postprocess boxes after transforms
        self.postprocess_annotations(
            input_data.images.image_sizes[0], input_data.targets
        )


    def sort_samples(
        self, input_samples: List[InputSample]
    ) -> List[InputSample]:
        """Sort samples according to sampling cfg."""
        if self.frame_order == "key_first":
            return input_samples
        if self.frame_order == "temporal":
            return sorted(
                input_samples,
                key=lambda x: x.metadata[0].frameIndex
                if x.metadata[0].frameIndex is not None
                else 0,
            )
        raise NotImplementedError(
            f"Frame ordering {self.frame_order} not " f"implemented."
        )

    def __getitem__(self, idx):
        """Wrap getitem to apply augmentations."""
        current_sample = super().__getitem__(idx)




class Vis4DDataModule(pl.LightningDataModule, metaclass=RegistryHolder):
    """Default Data module for Vis4D."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        samples_per_gpu: int,
        workers_per_gpu: int,
        train_datasets: Optional[List[ScalabelDataset]] = None,
        test_datasets: Optional[List[ScalabelDataset]] = None,
        predict_datasets: Optional[List[ScalabelDataset]] = None,
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
        train_dataset = data.ConcatDataset(self.train_datasets)
        if self.train_sampler is not None:
            batch_size, shuffle = 1, False
        else:
            batch_size, shuffle = self.samples_per_gpu, True
        train_dataloader = data.DataLoader(
            train_dataset,
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
            if get_world_size() > 1 and dataset.has_sequences:
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
