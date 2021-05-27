"""Visualizer class."""
import itertools
import os
from typing import List

import detectron2.utils.comm as comm
import torch
from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from PIL import Image
from scalabel.label.io import group_and_sort, save
from scalabel.label.typing import Frame

from openmt.struct import Boxes2D, InputSample

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
        self._predictions = []  # type: List[Frame]
        self._boxes2d = []  # type: List[Boxes2D]
        self._visualize = visualize

    def reset(self) -> None:
        """Preparation for a new round of evaluation."""
        self._predictions = []
        self._boxes2d = []

    def process(
        self, inputs: List[List[InputSample]], outputs: List[Boxes2D]
    ) -> None:
        """Process the pair of inputs and outputs."""
        for inp, out in zip(inputs, outputs):
            prediction = inp[0].metadata  # no ref views during test
            boxes2d = out.to(torch.device("cpu"))
            prediction.labels = boxes2d.to_scalabel(
                self._metadata.idx_to_class_mapping
            )
            boxes2d.metadata = {
                str(k): v
                for k, v in self._metadata.idx_to_class_mapping.items()
            }
            self._predictions.append(prediction)
            if self._visualize:
                self._boxes2d.append(boxes2d)

    def evaluate(self) -> None:
        """Evaluate the performance after processing all input/output pairs."""
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return  # pragma: no cover
        else:
            predictions = self._predictions  # pragma: no cover

        os.makedirs(os.path.join(self._output_dir), exist_ok=True)
        save(
            os.path.join(self._output_dir, "predictions.json"),
            predictions,
        )

        if self._visualize:
            os.makedirs(
                os.path.join(self._output_dir, "visualization"), exist_ok=True
            )
            has_videos = True
            for frame, box in zip(predictions, self._boxes2d):
                if frame.attributes is None:
                    frame.attributes = dict()  # pragma: no cover
                frame.attributes["boxs2d"] = box
                if frame.video_name is None:
                    frame.video_name = ""
                    has_videos = False

            if has_videos:
                predictions = group_and_sort(predictions)
            else:
                predictions = [predictions]
            for video_predictions in predictions:
                images = [Image.open(frame.url) for frame in video_predictions]
                boxes2d = [
                    frame.attributes["boxs2d"] for frame in video_predictions
                ]  # type: List[Boxes2D]
                images = draw_sequence(images, boxes2d)
                for img, frame in zip(images, video_predictions):
                    img.save(
                        os.path.join(
                            self._output_dir,
                            "visualization",
                            frame.video_name,
                            frame.name,
                        )
                    )
