"""DataPipe wraps datasets to share the prepossessing pipeline."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable

from torch.utils.data import ConcatDataset, Dataset

from .reference import MultiViewDataset
from .transforms.base import TFunctor
from .typing import DictData, DictDataOrList


class DataPipe(ConcatDataset[DictDataOrList]):
    """DataPipe class.

    This class wraps one or multiple instances of a PyTorch Dataset so that the
    preprocessing steps can be shared across those datasets. Composes dataset
    and the preprocessing pipeline.
    """

    def __init__(
        self,
        datasets: Dataset[DictDataOrList] | Iterable[Dataset[DictDataOrList]],
        preprocess_fn: Callable[
            [list[DictData]], list[DictData]
        ] = lambda x: x,
    ):
        """Creates an instance of the class.

        Args:
            datasets (Dataset | Iterable[Dataset]): Dataset(s) to be wrapped by
                this data pipeline.
            preprocess_fn (Callable[[list[DictData]], list[DictData]]):
                Preprocessing function of a single sample. It takes a list of
                samples and returns a list of samples. Defaults to identity
                function.
        """
        if isinstance(datasets, Dataset):
            datasets = [datasets]
        super().__init__(datasets)
        self.preprocess_fn = preprocess_fn

        self.has_reference = any(
            _check_reference(dataset) for dataset in datasets
        )

        if self.has_reference and not all(
            _check_reference(dataset) for dataset in datasets
        ):
            raise ValueError(
                "All datasets must be MultiViewDataset / has reference if "
                + "one of them is."
            )

    def __getitem__(self, idx: int) -> DictDataOrList:
        """Wrap getitem to apply augmentations."""
        samples = super().__getitem__(idx)
        if isinstance(samples, list):
            return self.preprocess_fn(samples)

        return self.preprocess_fn([samples])[0]


class MultiSampleDataPipe(DataPipe):
    """MultiSampleDataPipe class.

    This class wraps DataPipe to support augmentations that require multiple
    images (e.g., Mosaic and Mixup) by sampling additional indices for each
    image. NUM_SAMPLES needs to be defined as a class attribute for transforms
    that require multi-sample augmentation.
    """

    def __init__(
        self,
        datasets: Dataset[DictDataOrList] | Iterable[Dataset[DictDataOrList]],
        preprocess_fn: list[list[TFunctor]],
    ):
        """Creates an instance of the class.

        Args:
            datasets (Dataset | Iterable[Dataset]): Dataset(s) to be wrapped by
                this data pipeline.
            preprocess_fn (list[list[TFunctor]]): Preprocessing functions of a
                single sample. Different than DataPipe, this is a list of lists
                of transformation functions. The inner list is for transforms
                that needs to share the same sampled indices (e.g.,
                GenMosaicParameters and MosaicImages), and the outer list is
                for different transforms.
        """
        super().__init__(datasets)
        self.preprocess_fns = preprocess_fn

    def _sample_indices(self, idx: int, num_samples: int) -> list[int]:
        """Sample additional indices for multi-sample augmentation."""
        indices = [idx]
        for _ in range(1, num_samples):
            indices.append(random.randint(0, len(self) - 1))
        return indices

    def __getitem__(self, idx: int) -> DictDataOrList:
        """Wrap getitem to apply augmentations."""
        samples = super(DataPipe, self).__getitem__(idx)
        if not isinstance(samples, list):
            samples = [samples]
            single_view = True
        else:
            single_view = False

        for preprocess_fn in self.preprocess_fns:
            if hasattr(preprocess_fn[0], "NUM_SAMPLES"):
                num_samples = preprocess_fn[0].NUM_SAMPLES
                aug_inds = self._sample_indices(idx, num_samples)
                add_samples = [
                    super(DataPipe, self).__getitem__(ind)
                    for ind in aug_inds[1:]
                ]
                prep_samples = []
                for i, samp in enumerate(samples):
                    prep_samples.append(samp)
                    prep_samples += [
                        s[i] if isinstance(s, list) else s for s in add_samples
                    ]
            else:
                num_samples = 1
                prep_samples = samples
            for prep_fn in preprocess_fn:
                prep_samples = prep_fn.apply_to_data(prep_samples)  # type: ignore # pylint: disable=line-too-long
            samples = prep_samples[::num_samples]
        return samples[0] if single_view else samples


def _check_reference(dataset: Dataset[DictDataOrList]) -> bool:
    """Check if the datasets have reference."""
    has_reference = (
        dataset.has_reference if hasattr(dataset, "has_reference") else False
    )
    return has_reference or isinstance(dataset, MultiViewDataset)
