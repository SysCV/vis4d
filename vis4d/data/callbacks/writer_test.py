"""Writer callback tests."""
import os
import shutil
import unittest

from scalabel.label.io import load
from scalabel.label.utils import compare_results

from ..dataset import ScalabelDataset
from ..datasets import Scalabel
from .writer import DefaultWriterCallback


class TestDefaultWriterCallback(unittest.TestCase):
    """Test cases for DefaultWriterCallback."""

    write_path = "./unittests/writer_test/"

    def test_write(self) -> None:
        """Test write."""
        base_dir = "vis4d/engine/testcases/track/bdd100k-samples"
        dataset_loader = Scalabel(
            "test_dataset",
            f"{base_dir}/images",
            f"{base_dir}/labels",
            config_path=f"{base_dir}/config.toml",
            eval_metrics=["detect"],
        )
        dataset = ScalabelDataset(dataset_loader, False)
        writer = DefaultWriterCallback(0, dataset_loader, self.write_path)

        frames = []
        for samples in dataset:
            frame = samples[0].metadata[0]
            assert frame.labels is not None
            for label in frame.labels:
                label.score = 1.0
            writer.process(samples, {"detect": [frame.labels]})
            frames.append(frame)

        writer.write()
        pred_path = f"{self.write_path}/detect/detect_predictions.json"
        saved_frames = load(pred_path).frames
        compare_results(saved_frames, frames)

    @classmethod
    def tearDownClass(cls) -> None:
        """Class teardown."""
        if os.path.exists(cls.write_path):
            shutil.rmtree(cls.write_path)
