"""Load and convert waymo labels to scalabel format."""
import os

from scalabel.label.io import load, save
from scalabel.label.typing import Dataset

from vis4d.struct import ArgsType

from .base import BaseDatasetLoader



class Waymo(BaseDatasetLoader):  # pragma: no cover
    """Waymo Open dataloading class."""

    def __init__(
        self,
        input_dir: str,
        *args: ArgsType,
        save_images: bool = True,
        use_lidar_labels: bool = False,
        **kwargs: ArgsType
    ):
        """Init dataset loader."""
        self.input_dir = input_dir
        self.save_images = save_images
        self.use_lidar_labels = use_lidar_labels
        super().__init__(*args, **kwargs)

    def load_dataset(self) -> Dataset:
        """Convert Waymo annotations to Scalabel format."""
        assert (
            WAYMO_AVAILABLE
        ), "Using waymo dataset needs waymo open dataset reader installed!."

        # annotations is the path to the label file in scalabel format.
        # It's an optional attribute. When passed, if the file exists load it,
        # else create it to that location
        if self.annotations:
            scalabel_anns_path = self.annotations
        else:
            scalabel_anns_path = os.path.join(
                self.data_root, "scalabel_anns.json"
            )

        if not os.path.exists(scalabel_anns_path):
            # Read labels from tfrecords and save them to scalabel format
            dataset = from_waymo(
                self.input_dir,
                self.data_root,
                self.save_images,
                self.use_lidar_labels,
                self.num_processes,
            )
            save(scalabel_anns_path, dataset)
        else:
            # Load labels from existing file
            dataset = load(scalabel_anns_path, nprocs=self.num_processes)

        return dataset
