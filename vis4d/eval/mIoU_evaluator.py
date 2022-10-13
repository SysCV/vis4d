"""MIouEvaluator."""
from typing import Any, Dict, List, Tuple

import torch
from terminaltables import AsciiTable
from torchmetrics import ConfusionMatrix

from vis4d.common import MetricLogs, ModelOutput
from vis4d.common.typing import COMMON_KEYS
from vis4d.data.datasets.base import DictData

from .base import Evaluator


class MIouEvaluator(Evaluator):
    """Creates an evaluation to reports mIoU score and confusion matrix."""

    METRIC_MIOU = "mIoU"
    METRIC_CONFUSION_MATRIX = "confusion_matrix"
    METRIC_ALL = "all"

    def __init__(
        self,
        num_classes: int,
        class_mapping: Dict[int, str],
        semantics_pred_key: str = COMMON_KEYS.semantics3d,
        semantics_gt_key: str = COMMON_KEYS.semantics3d,
    ):
        """Creates a new evaluator.

        Args:
            num_classes (int): Number of semantic classes
            class_mapping (int): Dict mapping each class_id to a readable name
            semantics_pred_key (str): Key to obtain the semantics from the model output
            semantics_gt_key (str): Key to obtain the gt semantics from the input struct

        """
        super().__init__()
        self.num_classes = num_classes
        self._confusion_matrix = ConfusionMatrix(num_classes=num_classes)
        self.semantics_pred_key = semantics_pred_key
        self.semantics_gt_key = semantics_gt_key
        self.class_mapping = class_mapping
        self.reset()

    @property
    def metrics(self) -> List[str]:
        """Supported metrics."""
        return [
            MIouEvaluator.METRIC_MIOU,
            MIouEvaluator.METRIC_CONFUSION_MATRIX,
        ]

    def reset(self) -> None:
        """Reset the saved predictions to start new round of evaluation."""
        self._confusion_matrix.reset()

    def process(self, inputs: DictData, outputs: ModelOutput) -> None:
        """Process sample and update confusion matrix.

        Requires the keys specified in the constructor to be present
        i.e. inputs[semantics_gt_key], and  utputs[semantics_pred_key] must
        be present.
        Args:
             inputs (DictData): Inputs as fed to model
             outputs (ModelOuput): Model output
        """
        targets = inputs[self.semantics_gt_key]
        predictions = outputs[self.semantics_pred_key]
        if self._confusion_matrix.device != targets.device:
            self._confusion_matrix = self._confusion_matrix.to(targets.device)

        for _, (pred, gt_label) in enumerate(zip(predictions, targets)):
            self._confusion_matrix.update(pred.ravel(), gt_label.ravel())

    def _get_class_name_for_idx(self, idx: int) -> str:
        return self.class_mapping.get(idx, f"class_{idx}")

    def _get_miou(self) -> Tuple[Dict[str, Any], str]:
        confusion = self._confusion_matrix.compute()
        tp = torch.diag(confusion)
        fp = torch.sum(confusion, dim=0) - tp
        fn = torch.sum(confusion, dim=1) - tp
        iou = tp / (tp + fn + fp) * 100
        mIoU = iou.mean().cpu().item()

        iou_class_str = ", ".join(
            f"{self._get_class_name_for_idx(idx)}: ({d:.3f}%)"
            for idx, d in enumerate(iou)
        )
        return {
            "IoU": list(iou.cpu().numpy()),
            "mIoU": mIoU,
        }, f"mIoU: {mIoU:.3f}%, [" + iou_class_str + "]"

    def _get_confusion_matrix(self) -> Tuple[Dict[str, Any], str]:
        headers = ["Confusion"] + [
            self._get_class_name_for_idx(i) for i in range(self.num_classes)
        ]
        table_data = (
            self._confusion_matrix.compute()
            / (torch.sum(self._confusion_matrix.compute(), dim=1))
        ).cpu()
        data = list(
            [f"Class_{idx}"] + list(d.numpy())
            for idx, d in enumerate(table_data)
        )
        table = AsciiTable([headers] + data)
        return {"confusion_matrix": table_data}, table.table

    def evaluate(self, metric: str) -> Tuple[MetricLogs, str]:
        """Evaluate predictions. Returns a dict containing the raw data and a
        short description string containing a readable result.

        Args:
            metric (str): Metric to use. See @property metric
        """
        metric_data, short_description = {}, ""
        if metric in [MIouEvaluator.METRIC_MIOU, MIouEvaluator.METRIC_ALL]:
            data, desc = self._get_miou()
            for key, value in data.items():
                metric_data[key] = value
            short_description += desc + "\n"

        if metric in [
            MIouEvaluator.METRIC_CONFUSION_MATRIX,
            MIouEvaluator.METRIC_ALL,
        ]:
            data, desc = self._get_confusion_matrix()
            for key, value in data.items():
                metric_data[key] = value

            short_description += desc + "\n"
        return metric_data, short_description
