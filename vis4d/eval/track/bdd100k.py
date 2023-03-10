"""Scalabel tracking evaluator."""
from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

from torch import Tensor

from vis4d.common.imports import BDD100K_AVAILABLE, SCALABEL_AVAILABLE
from vis4d.common.typing import MetricLogs
from vis4d.data.datasets.bdd100k import bdd100k_track_map

from ..base import Evaluator

if SCALABEL_AVAILABLE and BDD100K_AVAILABLE:
    from bdd100k.common.utils import load_bdd100k_config
    from bdd100k.label.to_scalabel import bdd100k_to_scalabel
    from scalabel.eval.mot import acc_single_video_mot, evaluate_track
    from scalabel.label.io import group_and_sort, load
    from scalabel.label.transforms import xyxy_to_box2d
    from scalabel.label.typing import Frame, Label


class BDD100KEvaluator(Evaluator):
    """BDD100K 2D tracking evaluation class."""

    inverse_track_map = {v: k for k, v in bdd100k_track_map.items()}

    def __init__(self, annotation_path: str) -> None:
        """Initialize the evaluator."""
        super().__init__()
        self.annotation_path = annotation_path
        self.frames: list[Frame] = []
        self.reset()

    def __repr__(self) -> str:
        """Concise representation of the dataset evaluator."""
        return "BDD100K Tracking Evaluator"

    @property
    def metrics(self) -> list[str]:
        """Supported metrics."""
        return ["track"]

    def gather(  # type: ignore
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

    def process(  # type: ignore # pylint: disable=arguments-differ
        self,
        frame_ids: list[int],
        data_names: list[str],
        video_names: list[str],
        boxes_list: list[Tensor],
        class_ids_list: list[Tensor],
        scores_list: list[Tensor],
        track_ids_list: list[Tensor],
    ) -> None:
        """Process tracking results."""
        for (
            frame_id,
            data_name,
            video_name,
            boxes,
            scores,
            class_ids,
            track_ids,
        ) in zip(
            frame_ids,
            data_names,
            video_names,
            boxes_list,
            scores_list,
            class_ids_list,
            track_ids_list,
        ):
            labels = []
            for box, score, class_id, track_id in zip(
                boxes, scores, class_ids, track_ids
            ):
                box2d = xyxy_to_box2d(*box.cpu().numpy().tolist())
                label = Label(
                    box2d=box2d,
                    category=self.inverse_track_map[int(class_id)],
                    score=float(score),
                    id=str(int(track_id)),
                )
                labels.append(label)
            frame = Frame(
                name=data_name,
                videoName=video_name,
                frameIndex=frame_id,
                labels=labels,
            )
            self.frames.append(frame)

    def evaluate(self, metric: str) -> tuple[MetricLogs, str]:
        """Evaluate the dataset."""
        if metric == "track":
            bdd100k_anns = load(self.annotation_path)
            frames = bdd100k_anns.frames
            bdd100k_cfg = load_bdd100k_config("box_track")
            scalabel_frames = bdd100k_to_scalabel(frames, bdd100k_cfg)
            results = evaluate_track(
                acc_single_video_mot,
                gts=group_and_sort(scalabel_frames),
                results=group_and_sort(self.frames),
                config=bdd100k_cfg.scalabel,
                nproc=0,
            )
        else:
            raise NotImplementedError

        return {}, str(results)
