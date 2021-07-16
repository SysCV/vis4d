"""Visualizer class."""
import copy
import os
from collections import defaultdict
from typing import Dict, List

import torch
from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils import comm
from PIL import Image
from scalabel.label.io import group_and_sort, save
from scalabel.label.typing import Frame

from openmt.engine.utils import gather_predictions
from openmt.struct import Boxes2D, InputSample, ModelOutput

from .track import draw_sequence


class ScalabelVisualizer(DatasetEvaluator):  # type: ignore
    """Run model on sequence and visualize & save output."""

    def __init__(
        self,
        dataset_name: str,
        output_dir: str,
        distributed: bool = True,
        visualize: bool = True,
    ) -> None:
        """Init."""
        self._distributed = distributed
        self._output_dir = output_dir
        self._metadata = MetadataCatalog.get(dataset_name)
        self._predictions = defaultdict(list)  # type: Dict[str, List[Frame]]
        self._visualize = visualize

    def reset(self) -> None:
        """Preparation for a new round of evaluation."""
        self._predictions = defaultdict(list)

    def process(
        self, inputs: List[List[InputSample]], outputs: ModelOutput
    ) -> None:
        """Process the pair of inputs and outputs."""
        for key, output in outputs.items():
            for inp, out in zip(inputs, output):
                prediction = copy.deepcopy(inp[0].metadata)
                boxes2d = out.to(torch.device("cpu"))
                assert isinstance(
                    boxes2d, Boxes2D
                ), "Only Boxes2D output support for visualization."
                prediction.labels = boxes2d.to_scalabel(
                    self._metadata.idx_to_class_mapping
                )
                boxes2d.metadata = {
                    str(k): v
                    for k, v in self._metadata.idx_to_class_mapping.items()
                }
                attr = (
                    prediction.attributes
                    if prediction.attributes is not None
                    else dict()
                )
                attr["boxes2d"] = boxes2d  # type: ignore
                prediction.attributes = attr
                self._predictions[key].append(prediction)

    def evaluate(self) -> None:
        """Evaluate the performance after processing all input/output pairs."""
        if self._distributed:
            predictions_dict = gather_predictions(self._predictions)
            if not comm.is_main_process():
                return  # pragma: no cover
        else:
            predictions_dict = self._predictions  # pragma: no cover

        os.makedirs(os.path.join(self._output_dir), exist_ok=True)
        for key, predictions in predictions_dict.items():
            # save predictions
            predictions_without_boxes2d = []
            for frame in predictions:
                frame_without_boxes2d = copy.deepcopy(frame)
                assert frame_without_boxes2d.attributes is not None
                frame_without_boxes2d.attributes.pop("boxes2d")
                predictions_without_boxes2d.append(frame_without_boxes2d)

            save(
                os.path.join(self._output_dir, f"{key}_predictions.json"),
                predictions_without_boxes2d,
            )

            # visualize predictions
            if self._visualize:
                has_videos = True
                for frame in predictions:
                    if frame.video_name is None:
                        has_videos = False

                if has_videos:
                    predictions_grouped = group_and_sort(predictions)
                else:
                    predictions_grouped = [predictions]
                for video_predictions in predictions_grouped:
                    images = [
                        Image.open(frame.url) for frame in video_predictions
                    ]
                    boxes2d = []  # type: List[Boxes2D]
                    image_paths = []
                    for frame in video_predictions:
                        assert frame.attributes is not None
                        boxes2d.append(frame.attributes["boxes2d"])  # type: ignore # pylint: disable=line-too-long
                        if frame.video_name is None:
                            frame.video_name = ""
                        image_paths.append(
                            os.path.join(
                                self._output_dir,
                                f"{key}_visualization",
                                frame.video_name,
                                frame.name,
                            )
                        )
                    for fp, img in zip(
                        image_paths, draw_sequence(images, boxes2d)
                    ):
                        os.makedirs(os.path.dirname(fp), exist_ok=True)
                        img.save(fp)
