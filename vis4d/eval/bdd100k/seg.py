"""BDD100K segmentation evaluator."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

import numpy as np

from vis4d.common.array import array_to_numpy
from vis4d.common.imports import BDD100K_AVAILABLE, SCALABEL_AVAILABLE
from vis4d.common.typing import ArrayLike, MetricLogs
from vis4d.data.datasets.bdd100k import bdd100k_seg_map

from ..base import Evaluator

if SCALABEL_AVAILABLE and BDD100K_AVAILABLE:
    from bdd100k.common.utils import load_bdd100k_config
    from bdd100k.label.to_scalabel import bdd100k_to_scalabel
    from scalabel.eval.sem_seg import evaluate_sem_seg
    from scalabel.label.io import load
    from scalabel.label.transforms import mask_to_rle
    from scalabel.label.typing import Frame, Label
else:
    raise ImportError("scalabel or bdd100k is not installed.")


class BDD100KSegEvaluator(Evaluator):
    """BDD100K segmentation evaluation class."""

    inverse_seg_map = {v: k for k, v in bdd100k_seg_map.items()}

    def __init__(self, annotation_path: str) -> None:
        """Initialize the evaluator."""
        super().__init__()
        self.annotation_path = annotation_path
        self.frames: list[Frame] = []

        bdd100k_anns = load(annotation_path)
        frames = bdd100k_anns.frames
        self.config = load_bdd100k_config("sem_seg")
        self.gt_frames = bdd100k_to_scalabel(frames, self.config)

        self.reset()

    def __repr__(self) -> str:
        """Concise representation of the dataset evaluator."""
        return "BDD100K Segmentation Evaluator"

    @property
    def metrics(self) -> list[str]:
        """Supported metrics."""
        return ["sem_seg"]

    def gather(  # type: ignore # pragma: no cover
        self, gather_func: Callable[[Any], Any]
    ) -> None:
        """Gather variables in case of distributed setting (if needed).

        Args:
            gather_func (Callable[[Any], Any]): Gather function.
        """
        all_preds = gather_func(self.frames)
        if all_preds is not None:
            self.frames = list(itertools.chain(*all_preds))

    def reset(self) -> None:
        """Reset the evaluator."""
        self.frames = []

    def process_batch(
        self, data_names: list[str], masks_list: list[ArrayLike]
    ) -> None:
        """Process tracking results."""
        masks_numpy = [array_to_numpy(m, None) for m in masks_list]  # to numpy
        for data_name, masks in zip(data_names, masks_numpy):
            labels = []
            for i, class_id in enumerate(np.unique(masks)):
                label = Label(
                    rle=mask_to_rle((masks == class_id).astype(np.uint8)),
                    category=self.inverse_seg_map[int(class_id)],
                    id=str(i),
                )
                labels.append(label)
            frame = Frame(name=data_name, labels=labels)
            self.frames.append(frame)

    def evaluate(self, metric: str) -> tuple[MetricLogs, str]:
        """Evaluate the dataset."""
        if metric == "sem_seg":
            results = evaluate_sem_seg(
                ann_frames=self.gt_frames,
                pred_frames=self.frames,
                config=self.config.scalabel,
                nproc=0,
            )
        else:
            raise NotImplementedError

        return {}, str(results)
