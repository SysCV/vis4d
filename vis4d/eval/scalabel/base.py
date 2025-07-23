"""Scalabel base evaluator."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

from vis4d.common.imports import SCALABEL_AVAILABLE
from vis4d.common.typing import MetricLogs
from vis4d.eval.base import Evaluator

if SCALABEL_AVAILABLE:
    from scalabel.label.io import load
    from scalabel.label.typing import Config, Frame
    from scalabel.label.utils import get_leaf_categories
else:
    raise ImportError("scalabel is not installed.")


class ScalabelEvaluator(Evaluator):
    """Scalabel base evaluation class."""

    def __init__(
        self, annotation_path: str, config: Config | None = None
    ) -> None:
        """Initialize the evaluator."""
        super().__init__()
        self.annotation_path = annotation_path
        self.frames: list[Frame] = []

        dataset = load(self.annotation_path, validate_frames=False)
        self.gt_frames = dataset.frames
        if config is not None:
            self.config: Config | None = config
        else:
            self.config = dataset.config
        if self.config is not None and self.config.categories is not None:
            categories = get_leaf_categories(self.config.categories)
            self.inverse_cat_map = {
                cat_id: cat.name for cat_id, cat in enumerate(categories)
            }
        else:
            self.inverse_cat_map = {}
        self.reset()

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

    def evaluate(self, metric: str) -> tuple[MetricLogs, str]:
        """Evaluate the dataset."""
        raise NotImplementedError
