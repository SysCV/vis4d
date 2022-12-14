"""Evaluator callback tests."""
import shutil

import pytest
from _pytest.fixtures import FixtureRequest

from vis4d.unittest.utils import generate_input_sample

from ..datasets import BDD100K, COCO, BaseDatasetLoader, Scalabel
from .evaluator import DefaultEvaluatorCallback

ALLOWED_TASKS = [
    "detect",
    "ins_seg",
    "track",
    "seg_track",
    "sem_seg",
    "pan_seg",
]


def create_dataloader(task: str) -> BaseDatasetLoader:
    """Load data, setup data pipeline."""
    base = f"vis4d/engine/testcases/{task}/bdd100k-samples"
    if task in ["detect", "track"]:
        dataset_loader: BaseDatasetLoader = Scalabel(
            f"bdd100k_{task}_sample",
            f"{base}/images",
            f"{base}/labels/",
            config_path=f"{base}/config.toml",
            eval_metrics=[task],
        )
    elif task == "seg_track":
        base = base.replace("seg_track", "track")
        dataset_loader = Scalabel(
            f"bdd100k_{task}_sample",
            f"{base}/images",
            f"{base}/labels/",
            config_path=f"{base}/config.toml",
            eval_metrics=[task],
        )
    elif task == "ins_seg":
        base = "vis4d/engine/testcases/detect/bdd100k-samples"
        dataset_loader = COCO(
            f"bdd100k_{task}_sample",
            f"{base}/images",
            f"{base}/annotation_coco.json",
            config_path=f"{base}/insseg_config.toml",
            eval_metrics=[task],
        )
    else:
        # elif task in ["sem_seg", "pan_seg"]:
        base = (
            base.replace(task, "segment")
            if task == "sem_seg"
            else base.replace(task, "panoptic")
        )
        dataset_loader = BDD100K(
            f"bdd100k_{task}_sample",
            f"{base}/images",
            f"{base}/labels/",
            config_path=task,
            eval_metrics=[task],
        )
    return dataset_loader


@pytest.mark.parametrize("task", ALLOWED_TASKS)
def test_evaluate(task: str) -> None:
    """Test evaluation."""
    dataset_loader = create_dataloader(task)
    evaluator = DefaultEvaluatorCallback(0, dataset_loader, "unittests")

    def my_log(key: str, value: float, rank_zero_only: bool) -> None:
        print(key, value, rank_zero_only)

    evaluator.log = my_log

    frames = dataset_loader.frames
    for frame in frames:
        assert frame.labels is not None
        for label in frame.labels:
            label.score = 1.0
    test_inputs = [
        [generate_input_sample(28, 28, 1, 4, frame_name=f"test_frame{i}")]
        for i in range(len(frames))
    ]
    for i, test_input in enumerate(test_inputs):
        test_input[0].metadata[0] = frames[i]
    evaluator.process(
        test_inputs, {task: [f.labels for f in frames if f.labels is not None]}
    )
    results = evaluator.evaluate(0)
    assert isinstance(results, dict)
    assert len(results) > 0


@pytest.fixture(scope="module", autouse=True)
def teardown(request: FixtureRequest) -> None:
    """Clean up test files."""

    def remove_test_dir() -> None:
        shutil.rmtree("./unittests/", ignore_errors=True)

    request.addfinalizer(remove_test_dir)
