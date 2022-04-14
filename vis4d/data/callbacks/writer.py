"""Visualizer class."""
import copy
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from scalabel.label.typing import Frame, FrameGroup
from scalabel.vis.label import LabelViewer, UIConfig

from vis4d.struct import InputSample, ModelOutput
from vis4d.struct.data import Images
from vis4d.vis.utils import preprocess_image

from ..datasets import BaseDatasetLoader
from ..utils import all_gather_predictions


class BaseWriterCallback(Callback):
    """Prediction writer base class."""

    def __init__(
        self, dataloader_idx: int, output_dir: str, collect: str = "cpu"
    ):
        """Init."""
        assert collect in ["cpu", "gpu"], f"Collect arg {collect} unknown."
        self._output_dir = output_dir
        self._predictions: Dict[str, List[Frame]] = defaultdict(list)
        self.collect = collect
        self.dataloader_idx = dataloader_idx

    def reset(self) -> None:
        """Preparation for a new round of evaluation."""
        self._predictions = defaultdict(list)

    def gather(self, pl_module: pl.LightningModule) -> None:
        """Gather accumulated data."""
        preds = all_gather_predictions(
            self._predictions, pl_module, self.collect
        )
        if preds is not None:
            self._predictions = preds

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
        self.gather(pl_module)
        if trainer.is_global_zero:
            self.write()
        self.reset()


class DefaultWriterCallback(BaseWriterCallback):
    """Run model and visualize & save output."""

    def __init__(
        self,
        dataloader_idx: int,
        dataset_loader: BaseDatasetLoader,
        output_dir: str,
        visualize: bool = True,
    ) -> None:
        """Init."""
        super().__init__(dataloader_idx, output_dir)
        self._visualize = visualize
        self.viewer: Optional[LabelViewer] = None
        self.save_func = dataset_loader.save_predictions

        if self._output_dir is not None:
            os.makedirs(self._output_dir, exist_ok=True)

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
                if self._visualize:
                    reset_viewer = metadata.frameIndex in [None, 0]
                    if isinstance(prediction, FrameGroup):
                        rank_zero_warn(  # pragma: no cover
                            "Visualization don't support multi-sensor dataset."
                        )
                    else:
                        save_dir = os.path.join(
                            self._output_dir, f"{key}_visualization"
                        )
                        self.do_visualization(
                            metadata,
                            prediction,
                            save_dir,
                            inp[0].images,
                            reset_viewer=reset_viewer,
                        )

    def do_visualization(
        self,
        metadata: Frame,
        prediction: Frame,
        save_dir: str,
        images: Images,
        reset_viewer: bool,
    ) -> None:
        """Do Visualization."""
        if self.viewer is None or reset_viewer:
            size = metadata.size
            assert size is not None
            w, h = size.width, size.height
            self.viewer = LabelViewer(UIConfig(width=w, height=h))
        video_name = (
            prediction.videoName if prediction.videoName is not None else ""
        )
        save_path = os.path.join(
            save_dir,
            video_name,
            prediction.name,
        )
        self.viewer.draw(
            np.array(preprocess_image(images.tensor[0])),
            prediction,
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.viewer.save(save_path)

    def write(self) -> None:
        """Write the aggregated output."""
        for key, predictions in self._predictions.items():
            output_dir = os.path.join(self._output_dir, key)
            os.makedirs(output_dir, exist_ok=True)
            self.save_func(output_dir, key, predictions)
