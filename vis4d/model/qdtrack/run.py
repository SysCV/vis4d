"""QDTrack runtime configuration."""
from projects.common.datasets import bdd100k_track_map, mot_map
from projects.common.models import build_faster_rcnn, build_yolox
from projects.common.optimizers import sgd, step_schedule
from projects.qdtrack.data import QDTrackDataModule
from projects.qdtrack.qdtrack import QDTrackYOLOX
from vis4d.common.bbox.matchers import MaxIoUMatcher
from vis4d.common.bbox.poolers import MultiScaleRoIAlign
from vis4d.engine.trainer import BaseCLI
from vis4d.model import QDTrack
from vis4d.model.track.graph import QDTrackGraph
from vis4d.model.track.similarity import QDSimilarityHead


def setup_model(
    experiment: str,
    lr: float = 0.02,
    max_epochs: int = 12,
    detector: str = "FRCNN",
) -> QDTrack:
    """Setup model with experiment specific hyperparameters."""
    if experiment == "mot17":
        if detector == "YOLOX":
            track_graph = QDTrackGraph(
                keep_in_memory=30, init_score_thr=0.7, obj_score_thr=0.15
            )
        else:
            track_graph = QDTrackGraph(
                keep_in_memory=30, init_score_thr=0.9, obj_score_thr=0.5
            )
        category_mapping = mot_map
    elif experiment == "mot20":
        if detector == "YOLOX":
            track_graph = QDTrackGraph(
                keep_in_memory=30, init_score_thr=0.8, obj_score_thr=0.2
            )
        else:
            track_graph = QDTrackGraph(
                keep_in_memory=30, init_score_thr=0.9, obj_score_thr=0.5
            )
        category_mapping = mot_map
    elif experiment == "bdd100k":
        track_graph = QDTrackGraph(keep_in_memory=10)
        category_mapping = bdd100k_track_map
    else:
        raise NotImplementedError(f"Experiment {experiment} not known!")

    if detector == "YOLOX":
        detector = build_yolox(category_mapping)
        similarity_head = QDSimilarityHead(
            proposal_pooler=MultiScaleRoIAlign(0, (7, 7), [8, 16, 32]),
            in_features=["out0", "out1", "out2"],
            in_dim=320,
            proposal_matcher=MaxIoUMatcher(
                [0.5, 0.7], [0, -1, 1], allow_low_quality_matches=False
            )
            if experiment == "mot17"
            else None,
        )
        model = QDTrackYOLOX(
            category_mapping=category_mapping,
            detection=detector,
            similarity=similarity_head,
            track_graph=track_graph,
            lr_scheduler_init=step_schedule(max_epochs),
            optimizer_init=sgd(lr, weight_decay=0.0005),
        )
    else:
        if experiment == "mot17":
            detector = build_faster_rcnn(
                category_mapping, backbone="r50_caffe_fpn"
            )
            detector.load_pretrained_weights(
                "mmdet://faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/faster"
                "_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth",
                strict=False,
            )
            similarity_head = QDSimilarityHead(
                proposal_matcher=MaxIoUMatcher(
                    [0.5, 0.7], [0, -1, 1], allow_low_quality_matches=False
                )
            )
        else:
            detector = build_faster_rcnn(category_mapping)
            similarity_head = QDSimilarityHead()

        model = QDTrack(
            image_channel_mode=detector.image_channel_mode,
            category_mapping=category_mapping,
            detection=detector,
            similarity=similarity_head,
            track_graph=track_graph,
            lr_scheduler_init=step_schedule(max_epochs),
            optimizer_init=sgd(lr),
        )

    if experiment == "mot17":
        model.detector.clip_bboxes_to_image = False
    return model


class QDTrackCLI(BaseCLI):
    """QDTrack CLI."""

    def add_arguments_to_parser(self, parser):
        """Link data and model experiment argument."""
        parser.link_arguments("data.experiment", "model.experiment")
        parser.link_arguments("model.max_epochs", "trainer.max_epochs")


if __name__ == "__main__":
    QDTrackCLI(
        model_class=setup_model,
        datamodule_class=QDTrackDataModule,
    )
