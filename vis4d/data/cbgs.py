"""Class-balanced Grouping and Sampling for 3D Object Detection.

Implementation of `Class-balanced Grouping and Sampling for Point Cloud 3D
Object Detection <https://arxiv.org/abs/1908.09492>`_.
"""
from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from vis4d.common.time import Timer
from vis4d.common.logging import rank_zero_info

from .const import CommonKeys as K
from .datasets.util import print_class_histogram, CacheMappingMixin
from .reference import MultiViewDataset
from .typing import DictDataOrList


# TODO: Support sensor selection.
class CBGSDataset(CacheMappingMixin, Dataset[DictDataOrList]):
    """Balance the number of scenes under different classes."""

    def __init__(
        self,
        dataset,
        class_map: dict[str, int],
        class_key: str = K.boxes3d_classes,
        ignore: int = -1,
    ) -> None:
        """Creates an instance of the class."""
        super().__init__()
        self.dataset = dataset
        self.has_reference = isinstance(dataset, MultiViewDataset)
        self.cat2id = dict(sorted(class_map.items(), key=lambda x: x[1]))
        self.class_key = class_key
        self.ignore = ignore

        t = Timer()
        (
            class_sample_indices,
            sample_frequencies,
        ) = self._get_class_sample_indices()

        self.sample_indices = self._get_sample_indices(class_sample_indices)
        rank_zero_info(
            f"Generating {len(self.sample_indices)} CBGS samples takes "
            + f"{t.time():.2f} seconds."
        )

        self._show_histogram(sample_frequencies)

    def _show_histogram(self, sample_frequencies) -> None:
        """Show class histogram."""
        frequencies = {cat: 0 for cat in self.cat2id.keys()}

        for idx in self.sample_indices:
            freq = sample_frequencies[idx]
            for box3d_class in freq:
                frequencies[box3d_class] += freq[box3d_class]

        print_class_histogram(frequencies)

    def _get_class_sample_indices(self) -> dict[int, list[int]]:
        """Get sample indices."""
        class_sample_indices = {cat_id: [] for cat_id in self.cat2id.values()}
        sample_frequencies = []
        inv_class_map = {v: k for k, v in self.cat2id.items()}

        # Handle the case that dataset is already wrapped.
        if hasattr(self.dataset, "dataset"):
            dataset = self.dataset.dataset
        else:
            dataset = self.dataset

        samples_cat_ids = dataset.get_cat_ids()
        for idx, cat_ids in enumerate(samples_cat_ids):
            cur_cats = {cat_id: [] for cat_id in self.cat2id.values()}
            frequencies = {cat: 0 for cat in self.cat2id.keys()}

            for cat_id in cat_ids:
                cur_cats[cat_id] = [idx]
                frequencies[inv_class_map[cat_id]] += 1

            sample_frequencies.append(frequencies)
            for cat_id in cur_cats:
                class_sample_indices[cat_id] += cur_cats[cat_id]

        return class_sample_indices, sample_frequencies

    def _get_sample_indices(
        self, class_sample_indices: dict[int, list[int]]
    ) -> list[int]:
        """Load sample indices.

        Returns:
            list[dict]: List of indices after class sampling.
        """
        duplicated_samples = sum(
            [len(v) for _, v in class_sample_indices.items()]
        )
        class_distribution = {
            k: len(v) / duplicated_samples
            for k, v in class_sample_indices.items()
        }

        sample_indices = []

        frac = 1.0 / len(self.cat2id)
        ratios = [frac / v for v in class_distribution.values()]
        for cls_inds, ratio in zip(
            list(class_sample_indices.values()), ratios
        ):
            sample_indices += np.random.choice(
                cls_inds, int(len(cls_inds) * ratio)
            ).tolist()

        return sample_indices

    def __len__(self) -> int:
        """Return the length of data infos.

        Returns:
            int: Length of data infos.
        """
        return len(self.sample_indices)

    def __getitem__(self, idx: int) -> DictDataOrList:
        """Get original dataset idx according to the given index.

        Args:
            idx (int): The index of self.sample_indices.

        Returns:
            DictDataOrList: Data of the corresponding index.
        """
        ori_index = self.sample_indices[idx]
        return self.dataset[ori_index]
