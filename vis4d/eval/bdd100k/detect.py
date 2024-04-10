"""BDD100K detection evaluator."""

from __future__ import annotations

from vis4d.common.imports import BDD100K_AVAILABLE, SCALABEL_AVAILABLE
from vis4d.eval.scalabel import ScalabelDetectEvaluator

if BDD100K_AVAILABLE and SCALABEL_AVAILABLE:
    from bdd100k.common.utils import load_bdd100k_config


class BDD100KDetectEvaluator(ScalabelDetectEvaluator):
    """BDD100K 2D detection evaluation class."""

    METRICS_DET = "Det"
    METRICS_INS_SEG = "InsSeg"

    def __init__(
        self,
        annotation_path: str,
        config_path: str,
        mask_threshold: float = 0.0,
    ) -> None:
        """Initialize the evaluator."""
        config = load_bdd100k_config(config_path)
        super().__init__(
            annotation_path=annotation_path,
            config=config.scalabel,
            mask_threshold=mask_threshold,
        )

    def __repr__(self) -> str:
        """Concise representation of the dataset evaluator."""
        return "BDD100K Detection Evaluator"
