"""Load and convert kitti labels to scalabel format."""
import os
import os.path as osp
from typing import Optional

from scalabel.label.from_kitti import from_kitti
from scalabel.label.io import load, save
from scalabel.label.typing import Dataset

from vis4d.struct import ArgsType

from .base import BaseDatasetLoader


class KITTI(BaseDatasetLoader):  # pragma: no cover
    """KITTI dataloading class."""

    def __init__(
        self,
        *args: ArgsType,
        split: Optional[str] = None,
        data_type: Optional[str] = None,
        **kwargs: ArgsType
    ):
        """Init dataset loader."""
        self.data_type = data_type
        self.split = split
        super().__init__(*args, **kwargs)

    def load_dataset(self) -> Dataset:
        """Convert kitti annotations to Scalabel format."""
        assert self.annotations is not None
        if not os.path.exists(self.annotations):
            assert self.data_type is not None
            assert self.split is not None
            data_dir = osp.join(self.data_root, self.data_type, self.split)
            dataset = from_kitti(data_dir, self.data_type)
            save(self.annotations, dataset)
        else:
            dataset = load(
                self.annotations,
                validate_frames=self.validate_frames,
                nprocs=self.num_processes,
            )

        return dataset
