"""PanopticFPN runtime configuration."""
from projects.common.datasets import bdd100k_seg_map, bdd100k_track_map
from projects.common.models import build_mask_rcnn
from projects.common.optimizers import sgd, step_schedule
from projects.panoptic_fpn.data import PanopticFPNDataModule
from vis4d.engine.trainer import BaseCLI, DefaultTrainer
from vis4d.model.heads.dense_head import MMSegDecodeHead
from vis4d.model.heads.panoptic_head import SimplePanopticHead
from vis4d.model.panoptic import PanopticFPN


def setup_model(
    experiment: str, lr: float = 0.02, max_epochs: int = 12
) -> PanopticFPN:
    """Setup model with experiment specific hyperparameters."""
    if experiment != "bdd100k":
        raise NotImplementedError(f"Experiment {experiment} not known!")

    model = PanopticFPN(
        category_mapping=bdd100k_track_map,
        detection=build_mask_rcnn(bdd100k_track_map),
        seg_head=MMSegDecodeHead(
            mm_cfg=dict(
                type="FPNHead",
                in_channels=[256, 256, 256, 256],
                in_index=[0, 1, 2, 3],
                feature_strides=[4, 8, 16, 32],
                channels=128,
                num_classes=19,
                norm_cfg=dict(type="GN", num_groups=32, requires_grad=True),
                loss_decode=dict(type="CrossEntropyLoss", loss_weight=0.5),
                train_cfg={},
                test_cfg=dict(mode="whole"),
            ),
            category_mapping=bdd100k_seg_map,
        ),
        pan_head=SimplePanopticHead(
            ignore_class=[11, 12, 13, 14, 15, 16, 17, 18]
        ),
        lr_scheduler_init=step_schedule(max_epochs),
        optimizer_init=sgd(lr),
    )

    return model


class PanopticFPNCLI(BaseCLI):
    """PanopticFPN CLI."""

    def add_arguments_to_parser(self, parser):
        """Link data and model experiment argument."""
        parser.link_arguments("data.experiment", "model.experiment")
        parser.link_arguments("model.max_epochs", "trainer.max_epochs")


if __name__ == "__main__":
    PanopticFPNCLI(
        model_class=setup_model,
        datamodule_class=PanopticFPNDataModule,
        trainer_class=DefaultTrainer,
    )
