"""Visualizer class."""
import copy
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from scalabel.label.io import save
from scalabel.label.typing import Frame

from ..common.utils.distributed import get_rank, get_world_size
from ..struct import Boxes2D, Boxes3D, InputSample, ModelOutput
from ..vis.image import draw_image


class VisTWriterCallback(Callback):
    """VisT prediction writer base class."""

    def __init__(self, output_dir: str):
        """Init."""
        self._output_dir = output_dir
        self._predictions: Dict[str, List[Frame]] = defaultdict(list)

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


class ScalabelWriterCallback(VisTWriterCallback):
    """Run model and visualize & save output."""

    def __init__(
        self,
        output_dir: str,
        category_mapping: Optional[Dict[str, int]],
        visualize: bool = True,
    ) -> None:
        """Init."""
        super().__init__(output_dir)
        self._visualize = visualize
        self.cats_id2name: Optional[Dict[int, str]] = None
        if category_mapping is not None:
            self.cats_id2name = {v: k for k, v in category_mapping.items()}

    def process(
        self, inputs: List[List[InputSample]], outputs: ModelOutput
    ) -> None:
        """Process the pair of inputs and outputs."""
        for key, output in outputs.items():
            for inp, out in zip(inputs, output):
                prediction = copy.deepcopy(inp[0].metadata[0])
                out = out.to(torch.device("cpu"))  # type: ignore
                prediction.labels = out.to_scalabel(self.cats_id2name)
                self._predictions[key].append(prediction)

                if self._visualize:
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
<<<<<<< HEAD
                    assert isinstance(
                        out, Boxes2D
                    ), "Visualization only for boxes2d currently."
                    image = draw_image(inp[0].images.tensor[0], out)
=======
                    if isinstance(out, Boxes2D):
                        image = draw_image(
                            inp[0].images.tensor[0], boxes2d=out
                        )
                    elif isinstance(out, Boxes3D):
                        image = draw_image(
                            inp[0].images.tensor[0],
                            boxes3d=out,
                            intrinsics=inp[0].intrinsics,
                        )
                    else:
                        raise ValueError(f"Unknown result type: f{type(out)}")
>>>>>>> main
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    image.save(save_path)

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
