"""Example for dynamic api usage."""
from typing import Dict, List, Optional, Tuple

import torch

from vis4d import config
from vis4d.data.datasets import BaseDatasetConfig
from vis4d.engine.trainer import train
from vis4d.model import BaseModel, BaseModelConfig
from vis4d.model.backbone import BaseBackboneConfig, build_backbone
from vis4d.model.heads.dense_head import BaseDenseHeadConfig, build_dense_head
from vis4d.model.heads.roi_head import BaseRoIHeadConfig, build_roi_head
from vis4d.model.optimize import BaseOptimizerConfig
from vis4d.model.track.graph import TrackGraphConfig, build_track_graph
from vis4d.model.track.similarity import (
    SimilarityLearningConfig,
    build_similarity_head,
)
from vis4d.model.track.utils import split_key_ref_inputs
from vis4d.struct import (
    Boxes2D,
    Images,
    InputSample,
    LabelInstances,
    LossesType,
    ModelOutput,
)


class MyModelConfig(BaseModelConfig, extra="allow"):
    """My model config."""

    backbone: BaseBackboneConfig
    rpn_head: BaseDenseHeadConfig
    roi_head: BaseRoIHeadConfig
    similarity_head: SimilarityLearningConfig
    track_graph: TrackGraphConfig
    segmentation_head: BaseDenseHeadConfig


class MyModel(BaseModel):
    """Example qdtrack model."""

    def __init__(self, cfg: BaseModelConfig) -> None:
        """Init."""
        super().__init__(cfg)
        self.cfg: MyModelConfig = MyModelConfig(**cfg.dict())

        self.backbone = build_backbone(self.cfg.backbone)
        self.rpn_head = build_dense_head(self.cfg.rpn_head)
        self.roi_head = build_roi_head(self.cfg.roi_head)
        self.similarity_head = build_similarity_head(self.cfg.similarity_head)
        self.track_graph = build_track_graph(self.cfg.track_graph)

        self.segmentation_head = build_dense_head(self.cfg.segmentation_head)
        assert self.cfg.category_mapping is not None
        self.cat_mapping = {v: k for k, v in self.cfg.category_mapping.items()}

    def forward_train(self, batch_inputs: List[InputSample]) -> LossesType:
        """Forward pass during training stage."""
        key_inputs, ref_inputs = split_key_ref_inputs(batch_inputs)
        key_targets, ref_targets = key_inputs.targets, [
            x.targets for x in ref_inputs
        ]

        key_x = self.backbone(key_inputs)
        ref_x = [self.backbone(inp) for inp in ref_inputs]
        rpn_loss, key_proposals = self.rpn_head(key_inputs, key_x, key_targets)
        ref_proposals = [
            self.rpn_head(ref_inp, ref_x)
            for ref_inp, ref_x in zip(ref_inputs, ref_x)
        ]
        roi_loss, _ = self.roi_head(
            key_inputs, key_proposals, key_x, key_targets
        )
        track_loss, _ = self.similarity_head(
            [key_inputs, *ref_inputs],
            [key_proposals, *ref_proposals],
            [key_x, *ref_x],
            [key_targets, *ref_targets],
        )

        seg_loss, _ = self.segmentation_head(key_inputs, key_x, key_targets)
        return {**rpn_loss, **roi_loss, **track_loss, **seg_loss}

    def forward_test(self, batch_inputs: List[InputSample]) -> ModelOutput:
        """Forward pass during testing stage."""
        assert len(batch_inputs) == 1, "No reference views during test!"
        assert len(batch_inputs[0]) == 1, "Currently only BS=1 supported!"
        inputs = batch_inputs[0]

        features = self.backbone(inputs)
        proposals = self.rpn_head(inputs, features)
        detections = self.roi_head(inputs, proposals, features)
        embeddings = self.similarity_head(inputs, detections, features)

        predictions = LabelInstances(detections)
        tracks = self.track_graph(inputs, predictions, embeddings=embeddings)

        tracks_ = (
            tracks.boxes2d[0]
            .to(torch.device("cpu"))
            .to_scalabel(self.cat_mapping)
        )
        outputs = {"track": [tracks_]}

        segmentations = self.segmentation_head(inputs, features)
        semantic_segms_ = (
            segmentations[0]
            .to(torch.device("cpu"))
            .to_scalabel(self.seg_head.cat_mapping)
        )
        outputs["sem_seg"] = [semantic_segms_]
        return outputs


if __name__ == "__main__":
    conf = config.Config(
        model=dict(
            type="MyModelConfig",
            category_mapping={
                "pedestrian": 0,
                "rider": 1,
                "car": 2,
                "truck": 3,
                "bus": 4,
                "train": 5,
                "motorcycle": 6,
                "bicycle": 7,
            },
            image_channel_mode="RGB",
            optimizer=BaseOptimizerConfig(lr=0.001),
            # TODO add component configs
        ),
        launch=config.Launch(samples_per_gpu=2, workers_per_gpu=0),
        train=[
            BaseDatasetConfig(
                name="bdd100k_sample_train",
                type="BDD100K",
                annotations="vis4d/engine/testcases/track/bdd100k-samples/"
                "labels",
                data_root="vis4d/engine/testcases/track/bdd100k-samples/"
                "images/",
                config_path="box_track",
                eval_metrics=["detect"],
            )
        ],
        test=[
            BaseDatasetConfig(
                name="bdd100k_sample_val",
                type="BDD100K",
                annotations="vis4d/engine/testcases/track/bdd100k-samples/"
                "labels",
                data_root="vis4d/engine/testcases/track/bdd100k-samples/"
                "images/",
                config_path="box_track",
                eval_metrics=["track"],
            )
        ],
    )

    # choose according to setup
    # CPU
    train(conf)

    # single GPU
    trainer_args = {"gpus": "0,"}  # add arguments for PyTorchLightning trainer
    train(conf, trainer_args)

    # multi GPU
    trainer_args = {"gpus": "0,1"}
    train(conf, trainer_args)
