"""Visualizer class."""
import copy
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_warn
from scalabel.label.io import save
from scalabel.label.typing import Frame, FrameGroup
from scalabel.vis.label import LabelViewer, UIConfig

from ..common.utils.distributed import get_rank, get_world_size
from ..struct import InputSample, ModelOutput
from ..vis.utils import preprocess_image


class Vis4DWriterCallback(Callback):
    """Vis4D prediction writer base class."""

    def __init__(self, dataloader_idx: int, output_dir: str):
        """Init."""
        self._output_dir = output_dir
        self._predictions: Dict[str, List[Frame]] = defaultdict(list)
        self.dataloader_idx = dataloader_idx

    def reset(self) -> None:
        """Preparation for a new round of evaluation."""
        self._predictions = defaultdict(list)

    def process(
        self, inputs: List[List[InputSample]], outputs: ModelOutput
    ) -> None:
        """Process the pair of inputs and outputs."""
        raise NotImplementedError

    def write(self) -> None:
        """Write the aggregated output."""
        raise NotImplementedError

    def on_predict_batch_end(  # type: ignore
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Hook for on_predict_batch_end."""
        if dataloader_idx == self.dataloader_idx:
            self.process(batch, outputs)

    def on_predict_epoch_end(  # type: ignore
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Sequence[Any],
    ) -> None:
        """Hook for on_predict_epoch_end."""
        self.write()
        self.reset()


class ScalabelWriterCallback(Vis4DWriterCallback):
    """Run model and visualize & save output."""

    def __init__(
        self,
        dataloader_idx: int,
        output_dir: str,
        visualize: bool = True,
    ) -> None:
        """Init."""
        super().__init__(dataloader_idx, output_dir)
        self._visualize = visualize
        self.viewer: Optional[LabelViewer] = None

    def process(
        self, inputs: List[List[InputSample]], outputs: ModelOutput
    ) -> None:
        """Process the pair of inputs and outputs."""
        for key, output in outputs.items():
            for inp, out in zip(inputs, output):
                metadata = inp[0].metadata[0]
                prediction = copy.deepcopy(metadata)
                prediction.labels = out
                self._predictions[key].append(prediction)
                if self._visualize and isinstance(prediction, FrameGroup):
                    rank_zero_warn(  # pragma: no cover
                        "Visualization not supported for multi-sensor datasets"
                    )
                elif self._visualize:
                    if self.viewer is None or metadata.frameIndex in [None, 0]:
                        size = metadata.size
                        assert size is not None
                        w, h = size.width, size.height
                        self.viewer = LabelViewer(UIConfig(width=w, height=h))

                    video_name = (
                        prediction.videoName
                        if prediction.videoName is not None
                        else ""
                    )
                    save_path = os.path.join(
                        self._output_dir,
                        f"{key}_visualization",
                        video_name,
                        prediction.name,
                    )
                    self.viewer.draw(
                        np.array(preprocess_image(inp[0].images.tensor[0])),
                        prediction,
                    )
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    self.viewer.save(save_path)

    def write(self) -> None:
        """Write the aggregated output."""
        for key, predictions in self._predictions.items():
            os.makedirs(os.path.join(self._output_dir, key), exist_ok=True)
            if get_world_size() > 1:
                filename = f"predictions_{get_rank()}.json"  # pragma: no cover
            else:
                filename = "predictions.json"
            save(
                os.path.join(self._output_dir, key, filename),
                predictions,
            )
