"""Quasi-dense instance similarity learning model."""
import pickle
from typing import Dict, List, Tuple, Union

import torch

from projects.common.data_pipelines import default as default_augs
from vis4d.common.module import build_module
from vis4d.model.base import BaseModel
from vis4d.model.detect import (
    BaseDetector,
    BaseOneStageDetector,
    BaseTwoStageDetector,
)
from vis4d.model.optimize import BaseLRScheduler, BaseOptimizer
from vis4d.model.track.graph import BaseTrackGraph
from vis4d.model.track.similarity import BaseSimilarityHead
from vis4d.model.track.utils import split_key_ref_inputs
from vis4d.model.utils import postprocess_predictions, predictions_to_scalabel
from vis4d.struct import (
    ArgsType,
    Boxes2D,
    FeatureMaps,
    InputSample,
    LabelInstances,
    LossesType,
    ModelOutput,
    ModuleCfg,
    TLabelInstance,
)


class QDTrack(BaseModel):
    """QDTrack model - quasi-dense instance similarity learning."""

    def __init__(
        self,
        detection: BaseDetector,
        similarity: BaseSimilarityHead,
        track_graph: BaseTrackGraph,
        *args: ArgsType,
        **kwargs: ArgsType,
    ) -> None:
        """Init."""
        super().__init__(*args, **kwargs)
        assert self.category_mapping is not None, "Need category mapping"
        self.detector = detection
        assert isinstance(
            self.detector, (BaseTwoStageDetector, BaseOneStageDetector)
        )
        self.similarity_head = similarity
        self.track_graph = track_graph
        self.cat_mapping = {v: k for k, v in self.category_mapping.items()}
        self.with_mask = getattr(self.detector, "with_mask", False)

    def _run_heads_train(
        self,
        key_inputs: InputSample,
        ref_inputs: List[InputSample],
        key_x: FeatureMaps,
        ref_x: List[FeatureMaps],
    ) -> Tuple[LossesType, List[Boxes2D], List[List[Boxes2D]]]:
        """Get detection and tracking losses."""
        key_targets, ref_targets = key_inputs.targets, [
            x.targets for x in ref_inputs
        ]

        key_proposals: List[Boxes2D]
        ref_proposals: List[List[Boxes2D]]
        if isinstance(self.detector, BaseTwoStageDetector):
            # proposal generation
            rpn_losses, key_proposals = self.detector.generate_proposals(
                key_inputs, key_x, key_targets
            )
            with torch.no_grad():
                ref_proposals = [
                    self.detector.generate_proposals(inp, x, tgt)[1]
                    for inp, x, tgt in zip(ref_inputs, ref_x, ref_targets)
                ]

            # roi head
            assert isinstance(self.detector, BaseTwoStageDetector)
            roi_losses, _ = self.detector.generate_detections(
                key_inputs,
                key_x,
                key_proposals,
                key_targets,
            )
            det_losses = {**rpn_losses, **roi_losses}
        else:
            # one-stage detector
            det_losses, key_proposals = self.detector.generate_detections(
                key_inputs, key_x, key_targets
            )
            assert key_proposals is not None
            ref_proposals = []
            with torch.no_grad():
                for inp, x, tgt in zip(ref_inputs, ref_x, ref_targets):
                    ref_p = self.detector.generate_detections(inp, x, tgt)[1]
                    assert ref_p is not None
                    ref_proposals.append(ref_p)

        # from vis4d.vis.track import imshow_bboxes
        # for ref_inp, ref_props in zip(ref_inputs, ref_proposals):
        #     for ref_img, ref_prop in zip(ref_inp.images, ref_props):
        #         _, topk_i = torch.topk(ref_prop.boxes[:, -1], 100)
        #         imshow_bboxes(ref_img.tensor[0], ref_prop[topk_i])

        # track head
        track_losses, _ = self.similarity_head(
            [key_inputs, *ref_inputs],
            [key_proposals, *ref_proposals],
            [key_x, *ref_x],
            [key_targets, *ref_targets],
        )
        return {**det_losses, **track_losses}, key_proposals, ref_proposals

    def _run_heads_test(
        self, inputs: InputSample, feat: FeatureMaps
    ) -> Tuple[ModelOutput, LabelInstances, List[torch.Tensor]]:
        """Get detections and tracks."""
        if isinstance(self.detector, BaseTwoStageDetector):
            # two-stage detector
            proposals = self.detector.generate_proposals(inputs, feat)
            detections, instance_segms = self.detector.generate_detections(
                inputs, feat, proposals
            )
        else:
            # one-stage detector
            detections = self.detector.generate_detections(inputs, feat)
            instance_segms = None

        # similarity head
        embeddings = self.similarity_head(inputs, detections, feat)

        outs: Dict[str, List[TLabelInstance]] = {"detect": [d.clone() for d in detections]}  # type: ignore # pylint: disable=line-too-long
        if instance_segms is not None:
            outs["ins_seg"] = [s.clone() for s in instance_segms]

        postprocess_predictions(
            inputs,
            outs,
            self.detector.clip_bboxes_to_image,
            self.detector.resolve_overlap,
        )
        outputs = predictions_to_scalabel(outs, self.cat_mapping)

        predictions = LabelInstances(
            detections,
            instance_masks=instance_segms
            if instance_segms is not None
            else None,
        )
        return outputs, predictions, embeddings

    def _track(
        self,
        inputs: InputSample,
        predictions: LabelInstances,
        embeddings: List[torch.Tensor],
    ) -> ModelOutput:
        """Associate detections, update track graph."""
        tracks = self.track_graph(inputs, predictions, embeddings=embeddings)
        outs: Dict[str, List[TLabelInstance]] = {"track": tracks.boxes2d}  # type: ignore # pylint: disable=line-too-long
        if self.with_mask:
            outs["seg_track"] = tracks.instance_masks

        postprocess_predictions(
            inputs, outs, self.detector.clip_bboxes_to_image
        )
        return predictions_to_scalabel(outs, self.cat_mapping)

    def forward_train(self, batch_inputs: List[InputSample]) -> LossesType:
        """Forward function for training."""
        key_inputs, ref_inputs = split_key_ref_inputs(batch_inputs)

        # from vis4d.vis.image import imshow_bboxes
        # for batch_i, key_inp in enumerate(key_inputs):
        #    imshow_bboxes(
        #        key_inp.images.tensor[0], key_inp.targets.boxes2d[0]
        #    )
        #    for ref_i, ref_inp in enumerate(ref_inputs):
        #        imshow_bboxes(
        #            ref_inp[batch_i].images.tensor[0],
        #            ref_inp[batch_i].targets.boxes2d[0],
        #        )

        # feature extraction
        key_x = self.detector.extract_features(key_inputs)
        ref_x = [self.detector.extract_features(inp) for inp in ref_inputs]

        losses, _, _ = self._run_heads_train(
            key_inputs, ref_inputs, key_x, ref_x
        )
        return losses

    def forward_test(self, batch_inputs: List[InputSample]) -> ModelOutput:
        """Compute model output during inference."""
        assert len(batch_inputs) == 1, "No reference views during test!"
        assert len(batch_inputs[0]) == 1, "Currently only BS=1 supported!"

        result_path = ""
        if self.inference_result_path is not None:
            frame_name = batch_inputs[0].metadata[0].name
            result_path = self.inference_result_path + "/" + frame_name

        if self.inference_result_path is None or not self.data_backend.exists(
            result_path
        ):
            feat = self.detector.extract_features(batch_inputs[0])
            outs, predictions, embeddings = self._run_heads_test(
                batch_inputs[0], feat
            )
            if self.inference_result_path is not None:
                predictions = predictions.to(torch.device("cpu"))
                embeddings = [e.to(torch.device("cpu")) for e in embeddings]
                self.data_backend.set(
                    result_path, pickle.dumps([predictions, embeddings])
                )
        else:
            outs = {}
            predictions, embeddings = pickle.loads(
                self.data_backend.get(result_path)
            )
            predictions = predictions.to(self.device)
            embeddings = [e.to(self.device) for e in embeddings]

        outs.update(self._track(batch_inputs[0], predictions, embeddings))
        return outs


from mmdet.core.bbox.assigners import SimOTAAssigner


class ClippedSimOTAAssigner(SimOTAAssigner):
    """Modified SimOTAAssigner to support boxes with center outside of img."""

    def __init__(self, h: int, w: int, *args, **kwargs) -> None:
        """Init."""
        super().__init__(*args, **kwargs)
        self.im_h, self.im_w = h, w

    def get_in_gt_and_in_center_info(self, priors, gt_bboxes):
        num_gt = gt_bboxes.size(0)

        repeated_x = priors[:, 0].unsqueeze(1).repeat(1, num_gt)
        repeated_y = priors[:, 1].unsqueeze(1).repeat(1, num_gt)
        repeated_stride_x = priors[:, 2].unsqueeze(1).repeat(1, num_gt)
        repeated_stride_y = priors[:, 3].unsqueeze(1).repeat(1, num_gt)

        # is prior centers in gt bboxes, shape: [n_prior, n_gt]
        l_ = repeated_x - gt_bboxes[:, 0]
        t_ = repeated_y - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - repeated_x
        b_ = gt_bboxes[:, 3] - repeated_y

        deltas = torch.stack([l_, t_, r_, b_], dim=1)
        is_in_gts = deltas.min(dim=1).values > 0
        is_in_gts_all = is_in_gts.sum(dim=1) > 0

        gt_cxs = torch.clamp(
            (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0, min=0, max=self.im_w
        )
        gt_cys = torch.clamp(
            (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0, min=0, max=self.im_h
        )

        ct_box_l = gt_cxs - self.center_radius * repeated_stride_x
        ct_box_t = gt_cys - self.center_radius * repeated_stride_y
        ct_box_r = gt_cxs + self.center_radius * repeated_stride_x
        ct_box_b = gt_cys + self.center_radius * repeated_stride_y

        cl_ = repeated_x - ct_box_l
        ct_ = repeated_y - ct_box_t
        cr_ = ct_box_r - repeated_x
        cb_ = ct_box_b - repeated_y

        ct_deltas = torch.stack([cl_, ct_, cr_, cb_], dim=1)
        is_in_cts = ct_deltas.min(dim=1).values > 0
        is_in_cts_all = is_in_cts.sum(dim=1) > 0

        # in boxes or in centers, shape: [num_priors]
        is_in_gts_or_centers = is_in_gts_all | is_in_cts_all

        # both in boxes and centers, shape: [num_fg, num_gt]
        is_in_boxes_and_centers = (
            is_in_gts[is_in_gts_or_centers, :]
            & is_in_cts[is_in_gts_or_centers, :]
        )
        return is_in_gts_or_centers, is_in_boxes_and_centers


class QDTrackYOLOX(QDTrack):
    def __init__(
        self,
        *args: ArgsType,
        no_aug_epochs: int = 10,
        im_hw: Tuple[int, int] = (800, 1440),
        **kwargs: ArgsType,
    ) -> None:
        """Init."""
        super().__init__(*args, **kwargs)
        self.no_aug_epochs = no_aug_epochs
        self.im_hw = im_hw
        if self.detector.mm_detector.bbox_head.train_cfg:
            assign_args = (
                self.detector.mm_detector.bbox_head.train_cfg.assigner
            )
            self.detector.mm_detector.bbox_head.assigner = (
                ClippedSimOTAAssigner(*im_hw, **assign_args)
            )

    def on_train_epoch_start(self):
        """In the last training epochs: add L1 loss, turn off augmentations."""
        if self.current_epoch == self.trainer.max_epochs - self.no_aug_epochs:
            self.detector.mm_detector.bbox_head.use_l1 = True
            self.train_dataloader.transformations = default_augs(self.im_hw)

    def configure_optimizers(
        self,
    ) -> Tuple[List[BaseOptimizer], List[BaseLRScheduler]]:
        """Configure optimizers and schedulers of model."""
        params = []
        for name, param in self.named_parameters():
            param_group = {"params": [param]}
            if "bias" in name or "norm" in name:
                param_group["weight_decay"] = 0.0
            params.append(param_group)
        optimizer = build_optimizer(params, self.optimizer_cfg)
        scheduler = build_lr_scheduler(optimizer, self.lr_scheduler_cfg)
        return [optimizer], [scheduler]
