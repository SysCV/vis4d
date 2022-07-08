"""Visualizer class."""
import copy
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_warn

from vis4d.struct import InputData, InputSample, ModelOutput
from vis4d.vis.utils import preprocess_image

from ..utils import all_gather_predictions


class BaseWriterCallback(Callback):  # TODO make abstract
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

    def write(self, inputs: List[InputData], outputs: ModelOutput) -> None:
        """Process the pair of inputs and outputs."""
        raise NotImplementedError

    def flush(self) -> None:
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
        dataset,
        output_dir: str,
        visualize: bool = True,
    ) -> None:
        """Init."""
        super().__init__(dataloader_idx, output_dir)
        self._visualize = visualize
        self._save_func = dataset.save_predictions
        self._vis_func = dataset.visualize_predictions

        if self._output_dir is not None:
            os.makedirs(self._output_dir, exist_ok=True)

    def write(self, inputs: List[InputData], outputs: ModelOutput) -> None:
        """Process the pair of inputs and outputs."""
        for key, output in outputs.items():
            for inp, out in zip(inputs, output):
                metadata = inp.metadata
                prediction = copy.deepcopy(metadata)
                prediction.labels = out
                self._predictions[key].append(prediction)
                if self._visualize:
                    reset_viewer = metadata.frameIndex in [None, 0]
                    save_dir = os.path.join(
                        self._output_dir, f"{key}_visualization"
                    )
                    self._vis_func(
                        metadata,
                        prediction,
                        save_dir,
                        inp[0].images,
                        reset_viewer=reset_viewer,
                    )

    def flush(self) -> None:
        """Write the aggregated output."""
        for key, predictions in self._predictions.items():
            output_dir = os.path.join(self._output_dir, key)
            os.makedirs(output_dir, exist_ok=True)
            self.save_func(output_dir, key, predictions)
