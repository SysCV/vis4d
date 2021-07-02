"""Quasi-dense instance similarity learning model."""

from typing import List, Tuple

import torch

from openmt.struct import Boxes2D, InputSample, LossesType, ModelOutput

from .base import BaseModel, BaseModelConfig, build_model
from .detect import BaseTwoStageDetector
from .track.graph import TrackGraphConfig, build_track_graph
from .track.losses import LossConfig, build_loss
from .track.similarity import SimilarityLearningConfig, build_similarity_head
from .track.utils import cosine_similarity, split_key_ref_inputs


class QDGeneralizedRCNNConfig(BaseModelConfig):
    """Config for quasi-dense tracking model."""

    detection: BaseModelConfig
    similarity: SimilarityLearningConfig
    track_graph: TrackGraphConfig
    losses: List[LossConfig]
    softmax_temp: float = -1.0


class QDGeneralizedRCNN(BaseModel):
    """Generalized R-CNN for quasi-dense instance similarity learning."""

    def __init__(self, cfg: BaseModelConfig) -> None:
        """Init."""
        super().__init__()
        self.cfg = QDGeneralizedRCNNConfig(**cfg.dict())
        self.detector = build_model(self.cfg.detection)
        assert isinstance(self.detector, BaseTwoStageDetector)
        self.similarity_head = build_similarity_head(self.cfg.similarity)
        self.track_graph = build_track_graph(self.cfg.track_graph)
        self.track_loss = build_loss(self.cfg.losses[0])
        self.track_loss_aux = build_loss(self.cfg.losses[1])

    @property
    def device(self) -> torch.device:
        """Get device where input should be moved to."""
        return self.detector.device

    def forward_train(
        self, batch_inputs: List[List[InputSample]]
    ) -> LossesType:
        """Forward function for training."""
        # split into key / ref pairs NxM input --> key: N, ref: Nx(M-1)
        key_inputs, ref_inputs = split_key_ref_inputs(batch_inputs)

        # group by ref views by sequence: Nx(M-1) --> (M-1)xN
        ref_inputs = [
            [ref_inputs[j][i] for j in range(len(ref_inputs))]
            for i in range(len(ref_inputs[0]))
        ]

        # prepare targets
        key_targets = [input.instances.to(self.device) for input in key_inputs]
        ref_targets = [
            [input.instances.to(self.device) for input in inputs]
            for inputs in ref_inputs
        ]

        # from openmt.vis.image import imshow_bboxes
        # for batch_i, key_inp in enumerate(key_inputs):
        #     imshow_bboxes(key_inp.image.tensor[0], key_targets[batch_i])
        #     for ref_i, ref_inp in enumerate(ref_inputs):
        #         imshow_bboxes(
        #             ref_inp[batch_i].image.tensor[0],
        #             ref_targets[ref_i][batch_i],
        #         )

        # prepare inputs
        key_images = self.detector.preprocess_image(key_inputs)
        ref_images = [
            self.detector.preprocess_image(inp) for inp in ref_inputs
        ]

        # feature extraction
        key_x = self.detector.extract_features(key_images)
        ref_x = [self.detector.extract_features(img) for img in ref_images]

        # proposal generation
        key_proposals, rpn_losses = self.detector.generate_proposals(
            key_images, key_x, key_targets
        )
        with torch.no_grad():
            ref_proposals = [
                self.detector.generate_proposals(img, x)[0]
                for img, x in zip(ref_images, ref_x)
            ]

        # bbox head
        _, roi_losses = self.detector.generate_detections(
            key_images,
            key_x,
            key_proposals,
            key_targets,
            compute_detections=False,
        )
        det_losses = {**rpn_losses, **roi_losses}

        # from openmt.vis.track import imshow_bboxes
        # for ref_imgs, ref_props in zip(ref_images, ref_proposals):
        #     for ref_img, ref_prop in zip(ref_imgs, ref_props):
        #         _, topk_i = torch.topk(ref_prop.boxes[:, -1], 100)
        #         imshow_bboxes(ref_img.tensor[0], ref_prop[topk_i])

        # track head
        key_embeddings, key_track_targets = self.similarity_head(
            key_images,
            key_x,
            key_proposals,
            key_targets,
            filter_negatives=True,
        )
        ref_track_targets, ref_embeddings = [], []
        for inp, x, proposal, target in zip(
            ref_images, ref_x, ref_proposals, ref_targets
        ):
            embeds, targets = self.similarity_head(inp, x, proposal, target)
            ref_embeddings += [embeds]
            ref_track_targets += [targets]

        # from openmt.vis.track import visualize_matches
        # for ref_i, ref_inp in enumerate(ref_inputs):
        #     key_imgs = [key_inputs[i].image.tensor[0]
        #                 for i in range(len(key_inputs))]
        #     ref_imgs = [ref_inp[i].image.tensor[0]
        #                 for i in range(len(key_inputs))]
        #     visualize_matches(key_imgs, ref_imgs, key_track_targets,
        #     ref_track_targets[ref_i])

        track_losses = self.tracking_loss(
            key_embeddings,
            key_track_targets,
            ref_embeddings,
            ref_track_targets,
        )

        return {**det_losses, **track_losses}

    def match(
        self,
        key_embeds: Tuple[torch.Tensor],
        ref_embeds: List[Tuple[torch.Tensor]],
    ) -> Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        """Match key / ref embeddings based on cosine similarity."""
        # for each reference view
        dists, cos_dists = [], []
        for ref_embed in ref_embeds:
            # for each batch element
            dists_curr, cos_dists_curr = [], []
            for key_embed, ref_embed_ in zip(key_embeds, ref_embed):
                dist = cosine_similarity(
                    key_embed,
                    ref_embed_,
                    normalize=False,
                    temperature=self.cfg.softmax_temp,
                )
                dists_curr.append(dist)
                if self.track_loss_aux is not None:
                    cos_dist = cosine_similarity(key_embed, ref_embed_)
                    cos_dists_curr.append(cos_dist)

            dists.append(dists_curr)
            cos_dists.append(cos_dists_curr)
        return dists, cos_dists

    @staticmethod
    def get_track_targets(
        key_targets: List[Boxes2D], ref_targets: List[List[Boxes2D]]
    ) -> Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        """Create tracking target tensors."""
        # for each reference view
        track_targets, track_weights = [], []
        for ref_target in ref_targets:
            # for each batch element
            curr_targets, curr_weights = [], []
            for key_target, ref_target_ in zip(key_targets, ref_target):
                assert (
                    key_target.track_ids is not None
                    and ref_target_.track_ids is not None
                )
                # target shape: len(key_target) x len(ref_target_)
                target = (
                    key_target.track_ids.view(-1, 1)
                    == ref_target_.track_ids.view(1, -1)
                ).int()
                weight = (target.sum(dim=1) > 0).float()
                curr_targets.append(target)
                curr_weights.append(weight)
            track_targets.append(curr_targets)
            track_weights.append(curr_weights)
        return track_targets, track_weights

    def tracking_loss(
        self,
        key_embeddings: Tuple[torch.Tensor],
        key_targets: List[Boxes2D],
        ref_embeddings: List[Tuple[torch.Tensor]],
        ref_targets: List[List[Boxes2D]],
    ) -> LossesType:
        """Calculate losses for tracking.

        Key inputs are of type List[Tensor/Boxes2D] (Lists are length N)
        Ref inputs are of type List[List[Tensor/Boxes2D]] where the lists
        are of length MxN.
        Where M is the number of reference views and N is the
        number of batch elements.
        """
        losses = dict()

        loss_track = torch.tensor(0.0).to(self.device)
        loss_track_aux = torch.tensor(0.0).to(self.device)
        dists, cos_dists = self.match(key_embeddings, ref_embeddings)
        track_targets, track_weights = self.get_track_targets(
            key_targets, ref_targets
        )
        # for each reference view
        for curr_dists, curr_cos_dists, curr_targets, curr_weights in zip(
            dists, cos_dists, track_targets, track_weights
        ):
            # for each batch element
            for _dists, _cos_dists, _targets, _weights in zip(
                curr_dists, curr_cos_dists, curr_targets, curr_weights
            ):
                if all(_dists.shape):
                    loss_track += self.track_loss(
                        _dists,
                        _targets,
                        _weights,
                        avg_factor=_weights.sum() + 1e-5,
                    )
                    if self.track_loss_aux is not None:
                        loss_track_aux += self.track_loss_aux(
                            _cos_dists, _targets
                        )

        num_pairs = len(dists) * len(dists[0])
        losses["track_loss"] = loss_track / num_pairs
        if self.track_loss_aux is not None:
            losses["track_loss_aux"] = loss_track_aux / num_pairs

        return losses

    def forward_test(
        self, batch_inputs: List[InputSample], postprocess: bool = True
    ) -> ModelOutput:
        """Forward function during inference."""
        assert len(batch_inputs) == 1, "Currently only BS=1 supported!"

        # init graph at begin of sequence
        frame_id = batch_inputs[0].metadata.frame_index
        if frame_id == 0:
            self.track_graph.reset()

        # detector
        image = self.detector.preprocess_image(batch_inputs)
        feat = self.detector.extract_features(image)
        proposals, _ = self.detector.generate_proposals(image, feat)
        detections, _ = self.detector.generate_detections(
            image, feat, proposals
        )
        assert detections is not None

        # from openmt.vis.image import imshow_bboxes
        # imshow_bboxes(image.tensor[0], detections)

        # similarity head
        embeddings, _ = self.similarity_head(image, feat, detections)
        if postprocess:
            ori_wh = (
                batch_inputs[0].metadata.size.width,  # type: ignore
                batch_inputs[0].metadata.size.height,  # type: ignore
            )
            self.postprocess(ori_wh, image.image_sizes[0], detections[0])

        # associate detections, update graph
        tracks = self.track_graph(detections[0], frame_id, embeddings[0])

        return dict(detect=detections, track=[tracks])
