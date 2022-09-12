"""QD-3DT runtime configuration."""
from typing import Optional

from vis4d.common_to_revise.bbox.matchers import MaxIoUMatcher
from vis4d.common_to_revise.bbox.poolers import MultiScaleRoIAlign
from vis4d.common_to_revise.bbox.samplers import CombinedSampler
from vis4d.common_to_revise.datasets import kitti_track_map, nuscenes_track_map
from vis4d.common_to_revise.models import build_faster_rcnn
from vis4d.common_to_revise.optimizers import sgd, step_schedule
from vis4d.engine_to_clean.trainer import BaseCLI
from vis4d.op import QD3DT
from vis4d.op.heads.roi_head import QD3DTBBox3DHead
from vis4d.op.optimize.warmup import LinearLRWarmup
from vis4d.op.track.graph import QD3DTrackGraph
from vis4d.op.track.similarity import QDSimilarityHead
from vis4d.qd_3dt.data import QD3DTDataModule


def setup_model(
    experiment: str,
    lr: float = 0.01,
    max_epochs: int = 12,
    backbone: str = "r50_fpn",
    lstm_ckpt: Optional[str] = None,
) -> QD3DTOptimizer:
    """Setup model with experiment specific hyperparameters."""
    if experiment == "kitti":
        category_mapping = kitti_track_map
    elif "nuscenes" in experiment:
        category_mapping = nuscenes_track_map
    else:
        raise NotImplementedError(f"Experiment {experiment} not known!")

    track_graph = QD3DTrackGraph(
        keep_in_memory=10,
        lstm_ckpt=lstm_ckpt,
    )

    detector_kwargs = {
        "rpn_head.anchor_generator.scales": [4, 8],
        "rpn_head.anchor_generator.ratios": [0.25, 0.5, 1.0, 2.0, 4.0],
        "rpn_head.loss_bbox.type": "SmoothL1Loss",
        "rpn_head.loss_bbox.beta": 0.111,
        "roi_head.bbox_head.type": "ConvFCBBoxHead",
        "roi_head.bbox_head.num_shared_convs": 4,
        "roi_head.bbox_head.num_shared_fcs": 2,
        "roi_head.bbox_head.loss_cls.loss_weight": 5.0,
        "roi_head.bbox_head.loss_bbox.type": "SmoothL1Loss",
        "roi_head.bbox_head.loss_bbox.beta": 0.111,
        "roi_head.bbox_head.loss_bbox.loss_weight": 5.0,
    }

    detector = build_faster_rcnn(
        category_mapping, backbone, model_kwargs=detector_kwargs
    )

    box3d_head = QD3DTBBox3DHead(
        len(category_mapping),
        proposal_pooler=MultiScaleRoIAlign(
            resolution=[7, 7], strides=[4, 8, 16, 32], sampling_ratio=0
        ),
        proposal_sampler=CombinedSampler(
            batch_size_per_image=512,
            positive_fraction=0.25,
            pos_strategy="instance_balanced",
            neg_strategy="iou_balanced",
        ),
        proposal_matcher=MaxIoUMatcher(
            thresholds=[0.5, 0.5],
            labels=[0, -1, 1],
            allow_low_quality_matches=False,
        ),
    )

    similarity_head = QDSimilarityHead()

    model = QD3DT(
        detection=detector,
        similarity=similarity_head,
        track_graph=track_graph,
        bbox_3d_head=box3d_head,
    )
    runtime = QD3DTOptimizer(
        model,
        lr_scheduler_init=step_schedule(max_epochs),
        optimizer_init=sgd(lr),
        lr_warmup=LinearLRWarmup(warmup_ratio=0.1, warmup_steps=1000),
    )
    return runtime


class QD3DTCLI(BaseCLI):
    """QD3DT CLI."""

    def add_arguments_to_parser(self, parser):
        """Link data and model experiment argument."""
        parser.link_arguments("data.experiment", "model.experiment")
        parser.link_arguments("model.max_epochs", "trainer.max_epochs")


if __name__ == "__main__":
    QD3DTCLI(
        model_class=setup_model,
        datamodule_class=QD3DTDataModule,
    )
