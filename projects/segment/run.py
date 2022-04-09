"""Two-stage detector runtime configuration."""
from typing import List, Optional, Union

from projects.common.datasets import bdd100k_seg_map
from projects.common.models import build_deeplabv3plus, build_semantic_fpn
from projects.common.optimizers import poly_schedule, sgd
from projects.segment.data import SegmentDataModule
from vis4d.engine.trainer import BaseCLI, DefaultTrainer
from vis4d.model.segment import MMEncDecSegmentor


def setup_model(
    experiment: str,
    lr: float = 0.01,
    max_steps: int = 40000,
    segmentor: str = "DeepLabv3+",
    gpus: Optional[Union[List[int], str, int]] = 1,
) -> MMEncDecSegmentor:
    """Setup model with experiment specific hyperparameters."""
    if experiment == "bdd100k":
        category_mapping = bdd100k_seg_map
    else:
        raise NotImplementedError(f"Experiment {experiment} not known!")

    model_kwargs = {
        "lr_scheduler_init": poly_schedule(max_steps),
        "optimizer_init": sgd(lr, weight_decay=0.0005),
    }
    if (
        gpus is None
        or (isinstance(gpus, list) and len(gpus) <= 1)
        or int(gpus) == 1  # type: ignore
    ):
        # use standard BN for single-gpu training
        model_kwargs["model_kwargs"] = {
            "backbone.norm_cfg.type": "BN",
            "decode_head.norm_cfg.type": "BN",
            "auxiliary_head.norm_cfg.type": "BN",
        }

    if segmentor == "DeepLabv3+":
        model = build_deeplabv3plus(
            category_mapping, model_kwargs=model_kwargs
        )
    elif segmentor == "SemanticFPN":
        model = build_semantic_fpn(category_mapping, model_kwargs=model_kwargs)
    else:
        raise NotImplementedError(f"Segmentor {segmentor} not known!")

    return model


class SegmentCLI(BaseCLI):
    """Segment CLI."""

    def add_arguments_to_parser(self, parser):
        """Link data and model experiment argument."""
        parser.link_arguments("data.experiment", "model.experiment")
        parser.link_arguments("model.max_steps", "trainer.max_steps")
        parser.link_arguments("trainer.gpus", "model.gpus")


if __name__ == "__main__":
    SegmentCLI(
        model_class=setup_model,
        datamodule_class=SegmentDataModule,
        trainer_class=DefaultTrainer,
    )
