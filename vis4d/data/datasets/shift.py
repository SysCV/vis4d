"""Dataset loader for SHIFT format."""
from pytorch_lightning.utilities.distributed import rank_zero_info
from scalabel.label.io import load
from scalabel.label.typing import Category, Config, Dataset, ImageSize
from tqdm import tqdm

from .base import BaseDatasetLoader


class SHIFTDataset(BaseDatasetLoader):
    """Scalabel dataloading class."""

    CLASS_LABELS = [
        Category(name="bicycle"),
        Category(name="car"),
        Category(name="motor"),
        Category(name="truck"),
    ]

    def load_dataset(self) -> Dataset:
        """Load Scalabel frames from json."""
        assert self.annotations is not None
        self.dataset = load(
            self.annotations,
            nprocs=self.num_processes,
        )
        metadata_cfg = Config(
            imageSize=ImageSize(width=1280, height=800),
            categories=SHIFTDataset.CLASS_LABELS,
        )
        assert metadata_cfg is not None
        self.dataset.config = metadata_cfg
        return self.dataset

    def edit_data_path(self, dataset: Dataset) -> None:
        """Add temporal url for depth images via filename replacement."""
        rank_zero_info("Converting depth map URLs...")
        num_frames = len(dataset.frames)
        with tqdm(total=num_frames) as pbar:
            for frame in dataset.frames:
                depth_url = str(frame.url)
                depth_url = depth_url.replace("data.hdf5", "depth.hdf5")
                depth_url = depth_url.replace("img_center.png", "depth.png")
                if frame.attributes is None:  # pragma: no cover
                    frame.attributes = {}
                frame.attributes["__depth_url__"] = depth_url
                pbar.update(1)
