"""DataModule tests."""
import shutil
import unittest
from typing import List, Optional, Tuple

from parameterized import parameterized
from pytest import MonkeyPatch

from vis4d.struct import ArgsType, Images, InputSample
from vis4d.unittest.utils import MockModel, _trainer_builder

from .dataset import ScalabelDataset
from .datasets import BDD100K, COCO, KITTI, BaseDatasetLoader, Scalabel
from .handler import BaseDatasetHandler
from .mapper import BaseSampleMapper
from .module import BaseDataModule
from .transforms import (
    KorniaAugmentationWrapper,
    KorniaColorJitter,
    KorniaRandomHorizontalFlip,
    MixUp,
    Mosaic,
    RandomCrop,
    Resize,
)

ALLOWED_TASKS = [
    "track",
    "detect",
    "insseg",
    "segment",
    "panoptic",
    "detect3d",
    "track3d",
]


class SampleDataModule(BaseDataModule):
    """Load sample data to test data pipelines."""

    def __init__(
        self,
        task: str,
        im_hw: Tuple[int, int],
        *args: ArgsType,
        **kwargs: ArgsType,
    ):
        """Init."""
        assert task in ALLOWED_TASKS
        self.task = task
        self.im_hw = im_hw
        super().__init__(*args, **kwargs)

    def create_datasets(self, stage: Optional[str] = None) -> None:
        """Load data, setup data pipeline."""
        if self.task in ["track", "detect"]:
            annotations = (
                f"vis4d/engine/testcases/{self.task}/bdd100k-samples/labels/"
            )
            data_root = (
                f"vis4d/engine/testcases/{self.task}/bdd100k-samples/images"
            )
            config_path = (
                f"vis4d/engine/testcases/{self.task}/"
                "bdd100k-samples/config.toml"
            )
            dataset_loader: BaseDatasetLoader = Scalabel(
                f"bdd100k_{self.task}_sample",
                data_root,
                annotations,
                config_path=config_path,
            )
        elif self.task == "insseg":
            annotations = (
                "vis4d/engine/testcases/detect/"
                "bdd100k-samples/annotation_coco.json"
            )
            data_root = "vis4d/engine/testcases/detect/bdd100k-samples/images"
            config_path = (
                "vis4d/engine/testcases/detect/"
                "bdd100k-samples/insseg_config.toml"
            )
            dataset_loader = COCO(
                f"bdd100k_{self.task}_sample",
                data_root,
                annotations,
                config_path=config_path,
            )
        elif self.task == "segment":
            annotations = (
                f"vis4d/engine/testcases/{self.task}/bdd100k-samples/labels"
            )
            data_root = (
                f"vis4d/engine/testcases/{self.task}/bdd100k-samples/images"
            )
            dataset_loader = BDD100K(
                f"bdd100k_{self.task}_sample",
                data_root,
                annotations,
                config_path="sem_seg",
            )
        elif self.task == "panoptic":
            annotations = (
                f"vis4d/engine/testcases/{self.task}/bdd100k-samples/labels/"
            )
            data_root = "vis4d/engine/testcases/segment/bdd100k-samples/images"
            dataset_loader = BDD100K(
                f"bdd100k_{self.task}_sample",
                data_root,
                annotations,
                config_path="pan_seg",
            )
        else:
            annotations = "vis4d/engine/testcases/track/kitti-samples/labels/"
            data_root = "vis4d/engine/testcases/track/kitti-samples/"
            dataset_loader = KITTI(
                "kitti_sample",
                data_root,
                annotations,
            )

        mapper = None

        transforms = []
        if self.task == "track":
            transforms += [
                Mosaic(out_shape=self.im_hw),
                MixUp(out_shape=self.im_hw),
            ]

        transforms += [
            KorniaAugmentationWrapper(
                prob=1.0,
                kornia_type="RandomAffine",
                kwargs={
                    "degrees": 10.0,
                    "translate": [0.1, 0.1],
                    "scale": [0.5, 1.5],
                    "shear": [2.0, 2.0],
                },
            ),
            KorniaRandomHorizontalFlip(prob=0.5),
            Resize(shape=self.im_hw, keep_ratio=True, scale_range=(0.8, 1.2)),
            RandomCrop(shape=self.im_hw),
            KorniaColorJitter(
                prob=0.5,
                kwargs={
                    "brightness": [0.875, 1.125],
                    "contrast": [0.5, 1.5],
                    "saturation": [0.5, 1.5],
                    "hue": [-0.1, 0.1],
                },
            ),
        ]

        if self.task == "insseg":
            targets: Tuple[str, ...] = ("boxes2d", "instance_masks")
            mapper = BaseSampleMapper(targets_to_load=targets)
        if self.task == "segment":
            targets = ("boxes2d", "semantic_masks")
            mapper = BaseSampleMapper(targets_to_load=targets)
        elif self.task == "panoptic":
            targets = ("boxes2d", "instance_masks", "semantic_masks")
            mapper = BaseSampleMapper(targets_to_load=targets)
        elif self.task == "detect3d":
            inputs: Tuple[str, ...] = ("images", "intrinsics", "pointcloud")
            targets = (
                "boxes2d",
                "boxes3d",
            )
            mapper = BaseSampleMapper(
                inputs_to_load=inputs, targets_to_load=targets
            )
        elif self.task == "track3d":
            inputs = ("images", "intrinsics", "extrinsics", "pointcloud")
            targets = (
                "boxes2d",
                "boxes3d",
            )
            mapper = BaseSampleMapper(
                inputs_to_load=inputs, targets_to_load=targets
            )

        self.train_datasets = BaseDatasetHandler(
            ScalabelDataset(dataset_loader, True, mapper=mapper),
            transformations=transforms,
        )
        self.test_datasets = [
            BaseDatasetHandler(
                ScalabelDataset(dataset_loader, False, mapper=mapper)
            )
        ]


class TestDataModule(unittest.TestCase):
    """Test cases for base data module."""

    predict_dir = (
        "vis4d/engine/testcases/track/bdd100k-samples/images/"
        "00091078-875c1f73/"
    )

    def setUp(self) -> None:
        """Set up test case."""
        self.trainer = _trainer_builder("data_module_test")
        self.model = MockModel(model_param=7)
        self.monkeypatch = MonkeyPatch()

    @parameterized.expand(ALLOWED_TASKS)  # type: ignore
    def test_data(self, task: str) -> None:
        """Test tracking data loading."""
        batch_size = 1
        im_hw = (360, 640)
        data_module = SampleDataModule(
            task,
            im_hw,
            input_dir=self.predict_dir,
            workers_per_gpu=0,
            samples_per_gpu=batch_size,
        )

        Images.stride = 1

        def train_step(batch: List[InputSample]) -> None:
            self.assertEqual(len(batch), 1)
            sample = batch[0]
            self.assertEqual(len(sample), batch_size)
            N, C, H, W = sample.images.tensor.shape
            self.assertEqual(N, batch_size)
            self.assertEqual(C, 3)
            self.assertLessEqual(H, im_hw[0])
            self.assertLessEqual(W, im_hw[1])

        self.monkeypatch.setattr(self.model, "training_step", train_step)

        # test full loop
        self.trainer.fit(self.model, data_module)
        self.trainer.test(self.model, data_module)
        self.trainer.predict(self.model, data_module)
        Images.stride = 32

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up test files."""
        shutil.rmtree("./unittests/", ignore_errors=True)
