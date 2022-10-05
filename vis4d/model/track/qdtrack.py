"""Quasi-dense instance similarity learning model."""
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from vis4d.data_to_revise.transforms import Resize
from vis4d.op.base.resnet import ResNet
from vis4d.op.detect.faster_rcnn import (
    FasterRCNNHead,
    FRCNNOut,
    get_default_anchor_generator,
    get_default_rcnn_box_encoder,
    get_default_rpn_box_encoder,
)
from vis4d.op.detect.faster_rcnn_test import identity_collate, normalize
from vis4d.op.detect.rcnn import DetOut, RCNNLoss, RCNNLosses, RoI2Det
from vis4d.op.detect.rpn import RPNLoss, RPNLosses
from vis4d.op.fpp.fpn import FPN
from vis4d.op.track.graph.assignment import TrackIDCounter
from vis4d.op.track.qdtrack import (
    QDSimilarityHead,
    QDTrackAssociation,
    QDTrackInstanceSimilarityLoss,
)
from vis4d.state.track.qdtrack import QDTrackMemory, QDTrackState

REV_KEYS = [
    (r"^detector.rpn_head.mm_dense_head\.", "rpn_head."),
    ("\.rpn_reg\.", ".rpn_box."),
    (r"^detector.roi_head.mm_roi_head.bbox_head\.", "roi_head."),
    (r"^detector.backbone.mm_backbone\.", "body."),
    (
        r"^detector.backbone.neck.mm_neck.lateral_convs\.",
        "inner_blocks.",
    ),
    (
        r"^detector.backbone.neck.mm_neck.fpn_convs\.",
        "layer_blocks.",
    ),
    ("\.conv.weight", ".weight"),
    ("\.conv.bias", ".bias"),
]


class QDTrack(nn.Module):
    """QDTrack model - quasi-dense instance similarity learning."""

    def __init__(
        self,
        memory_size: int = 10,
        num_ref_views: int = 1,
        proposal_append_gt: bool = True,
    ) -> None:
        """Init."""
        super().__init__()
        self.num_ref_views = num_ref_views
        self.similarity_head = QDSimilarityHead()

        # only in inference
        self.track_graph = QDTrackAssociation()
        self.track_memory = QDTrackMemory(memory_limit=memory_size)

        self.box_sampler = CombinedSampler(
            batch_size=256,
            positive_fraction=0.5,
            pos_strategy="instance_balanced",
            neg_strategy="iou_balanced",
        )

        self.box_matcher = MaxIoUMatcher(
            thresholds=[0.3, 0.7],
            labels=[0, -1, 1],
            allow_low_quality_matches=False,
        )
        self.proposal_append_gt = proposal_append_gt
        self.track_loss = QDTrackInstanceSimilarityLoss()

    def debug_logging(self, logger) -> Dict[str, torch.Tensor]:
        """Logging for debugging"""
        # from vis4d.vis.track import imshow_bboxes
        # for ref_inp, ref_props in zip(ref_inputs, ref_proposals):
        #     for ref_img, ref_prop in zip(ref_inp.images, ref_props):
        #         _, topk_i = torch.topk(ref_prop.boxes[:, -1], 100)
        #         imshow_bboxes(ref_img.tensor[0], ref_prop[topk_i])
        # for batch_i, key_inp in enumerate(key_inputs):
        #    imshow_bboxes(
        #        key_inp.images.tensor[0], key_inp.targets.boxes2d[0]
        #    )
        #    for ref_i, ref_inp in enumerate(ref_inputs):
        #        imshow_bboxes(
        #            ref_inp[batch_i].images.tensor[0],
        #            ref_inp[batch_i].targets.boxes2d[0],
        #        )

    def forward(
        self,
        features: List[torch.Tensor],
        det_boxes: List[torch.Tensor],
        det_scores: List[torch.Tensor],
        det_class_ids: List[torch.Tensor],
        frame_ids: Optional[Tuple[int, ...]] = None,
        target_boxes: Optional[List[torch.Tensor]] = None,
        target_track_ids: Optional[List[torch.Tensor]] = None,
    ) -> List[QDTrackState]:
        """Forward function."""
        if target_boxes is not None:
            assert (
                target_track_ids is not None
            ), "Need targets during training!"
            return self._forward_train(
                features,
                det_boxes,
                target_boxes,
                target_track_ids,
            )
        assert frame_ids is not None, "Need frame ids during inference!"
        return self._forward_test(
            features, det_boxes, det_scores, det_class_ids, frame_ids
        )

    def _split_views(
        self,
        embeddings: List[torch.Tensor],
        target_track_ids: List[torch.Tensor],
    ) -> Tuple[
        List[torch.Tensor],
        List[List[torch.Tensor]],
        List[torch.Tensor],
        List[List[torch.Tensor]],
    ]:
        """Split batch and reference view dimension."""
        B, R = len(embeddings), self.num_ref_views + 1
        key_embeddings = [embeddings[i] for i in range(0, B, R)]
        key_track_ids = [target_track_ids[i] for i in range(0, B, R)]
        ref_embeddings, ref_track_ids = [], []
        for i in range(1, B, R):
            current_refs, current_track_ids = [], []
            for j in range(i, i + R - 1):
                current_refs.append(embeddings[j])
                current_track_ids.append(target_track_ids[j])
            ref_embeddings.append(current_refs)
            ref_track_ids.append(current_track_ids)
        return key_embeddings, ref_embeddings, key_track_ids, ref_track_ids

    @torch.no_grad()
    def _sample_proposals(
        self,
        det_boxes: List[torch.Tensor],
        target_boxes: List[torch.Tensor],
        target_track_ids: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Sample proposals for instance similarity learning."""
        B, R = len(det_boxes), self.num_ref_views + 1

        if self.proposal_append_gt:
            det_boxes = [
                torch.cat([d, t]) for d, t in zip(det_boxes, target_boxes)
            ]

        (
            sampled_box_indices,
            sampled_target_indices,
            sampled_labels,
        ) = match_and_sample_proposals(
            self.box_matcher,
            self.box_sampler,
            det_boxes,
            target_boxes,
        )
        sampled_boxes, sampled_track_ids = [], []
        for i in range(B):
            positives = sampled_labels[i] == 1
            if i % R == 0:  # take only positives for keyframes
                sampled_box = det_boxes[i][sampled_box_indices[i]][positives]
                sampled_tr_id = target_track_ids[i][sampled_target_indices[i]][
                    positives
                ]
            else:  # set track_ids to -1 for all negatives
                sampled_box = det_boxes[i][sampled_box_indices[i]]
                sampled_tr_id = target_track_ids[i][sampled_target_indices[i]]
                sampled_tr_id[~positives] = -1

            sampled_boxes.append(sampled_box)
            sampled_track_ids.append(sampled_tr_id)
        return sampled_boxes, sampled_track_ids

    def _forward_train(
        self,
        features: List[torch.Tensor],
        det_boxes: List[torch.Tensor],
        target_boxes: List[torch.Tensor],
        target_track_ids: List[torch.Tensor],
    ):
        """TODO define return type."""
        sampled_boxes, sampled_track_ids = self._sample_proposals(
            det_boxes, target_boxes, target_track_ids
        )
        embeddings = self.similarity_head(features, sampled_boxes)
        return self.track_loss(
            *self._split_views(embeddings, sampled_track_ids)
        )

    def _forward_test(
        self,
        features: List[torch.Tensor],
        det_boxes: List[torch.Tensor],
        det_scores: List[torch.Tensor],
        det_class_ids: List[torch.Tensor],
        frame_ids: Tuple[int, ...],
    ) -> List[QDTrackState]:
        """Forward during test."""
        embeddings = self.similarity_head(features, det_boxes)

        batched_tracks = []
        for frame_id, box, score, cls_id, embeds in zip(
            frame_ids, det_boxes, det_scores, det_class_ids, embeddings
        ):
            # reset graph at begin of sequence
            if frame_id == 0:
                self.track_memory.reset()
                TrackIDCounter.reset()

            cur_memory = self.track_memory.get_current_tracks(box.device)
            track_ids, filter_indices = self.track_graph(
                box,
                score,
                cls_id,
                embeds,
                cur_memory.track_ids,
                cur_memory.class_ids,
                cur_memory.embeddings,
            )

            data = QDTrackState(
                track_ids,
                box[filter_indices],
                score[filter_indices],
                cls_id[filter_indices],
                embeds[filter_indices],
            )
            self.track_memory.update(data)
            batched_tracks.append(self.track_memory.last_frame)

        return batched_tracks


class QDTrackModel(nn.Module):
    """Wrap qdtrack with detector."""

    def __init__(self) -> None:
        """Init."""
        super().__init__()
        anchor_gen = get_default_anchor_generator()
        rpn_bbox_encoder = get_default_rpn_box_encoder()
        rcnn_bbox_encoder = get_default_rcnn_box_encoder()
        self.backbone = ResNet("resnet50", pretrained=True, trainable_layers=3)
        self.fpn = FPN(self.backbone.out_channels[2:], 256)
        self.faster_rcnn_heads = FasterRCNNHead(
            num_classes=8,
            anchor_generator=anchor_gen,
            rpn_box_encoder=rpn_bbox_encoder,
            rcnn_box_encoder=rcnn_bbox_encoder,
        )
        self.transform_detections = RoI2Det(rcnn_bbox_encoder)
        self.qdtracker = QDTrack()

    def forward(
        self,
        images: torch.Tensor,
        images_hw: List[Tuple[int, int]],
        frame_ids: List[int],
    ) -> List[QDTrackState]:
        """Forward."""
        return self._forward_test(images, images_hw, frame_ids)

    def _forward_test(
        self,
        images: torch.Tensor,
        images_hw: List[Tuple[int, int]],
        frame_ids: List[int],
    ) -> List[QDTrackState]:
        """Forward inference stage."""
        features = self.backbone(images)
        features = self.fpn(features)
        detector_out = self.faster_rcnn_heads(features, images_hw)

        boxes, scores, class_ids = self.transform_detections(
            *detector_out.roi, detector_out.proposals.boxes, images_hw
        )
        outs = self.qdtracker(features, boxes, scores, class_ids, frame_ids)
        return outs


class QDTrackCLI(BaseCLI):
    """Detect CLI."""

    def add_arguments_to_parser(self, parser):
        """Link data and model experiment argument."""
        parser.link_arguments("data.experiment", "model.experiment")
        parser.link_arguments("model.max_epochs", "trainer.max_epochs")


if __name__ == "__main__":
    """Example:

    python -m vis4d.model.detect.faster_rcnn fit --data.experiment coco --trainer.gpus 6,7 --data.samples_per_gpu 8 --data.workers_per_gpu 8"""
    DetectCLI(model_class=setup_model, datamodule_class=DetectDataModule)


# ## setup model
# qdtrack = QDTrackModel()
# qdtrack.to(device)

# optimizer = optim.SGD(qdtrack.parameters(), lr=learning_rate, momentum=0.9)
# scheduler = optim.lr_scheduler.MultiStepLR(
#     optimizer, milestones=[8, 11], gamma=0.1
# )

# ## setup datasets
# train_sample_mapper = BaseSampleMapper(skip_empty_samples=True)
# train_sample_mapper.setup_categories(bdd100k_track_map)
# train_transforms = default(train_resolution)
# ref_sampler = BaseReferenceSampler(
#     scope=3, num_ref_imgs=1, skip_nomatch_samples=True
# )

# train_data = BaseDatasetHandler(
#     [
#         ScalabelDataset(bdd100k_det_train(), True, train_sample_mapper),
#         ScalabelDataset(
#             bdd100k_track_train(), True, train_sample_mapper, ref_sampler
#         ),
#     ],
#     clip_bboxes_to_image=True,
#     transformations=train_transforms,
# )
# train_loader = DataLoader(
#     train_data,
#     batch_size=batch_size,
#     shuffle=True,
#     collate_fn=identity_collate,
#     num_workers=batch_size // 2,
# )

# val_loader = bdd100k_track_val()
# test_sample_mapper = BaseSampleMapper()
# test_sample_mapper.setup_categories(bdd100k_track_map)
# test_transforms = [Resize(shape=test_resolution, keep_ratio=True)]
# test_data = BaseDatasetHandler(
#     [ScalabelDataset(val_loader, False, test_sample_mapper)],
#     transformations=test_transforms,
# )
# test_loader = DataLoader(
#     test_data,
#     batch_size=1,
#     shuffle=False,
#     collate_fn=identity_collate,
#     num_workers=2,
# )

# ## validation loop
# @torch.no_grad()
# def validation_loop(model):
#     """Validate current model with test dataset."""
#     model.eval()
#     gts = []
#     preds = []
#     class_ids_to_name = {i: s for s, i in bdd100k_track_map.items()}
#     print("Running validation...")
#     for data in tqdm(test_loader):
#         data = InputSample.cat(data[0], device)
#         images = data.images.tensor
#         output_wh = data.images.image_sizes
#         frame_ids = [metadata.frameIndex for metadata in data.metadata]

#         outs = model(
#             normalize(images), [(h, w) for w, h in output_wh], frame_ids
#         )

#         for i, (metadata, out, wh) in enumerate(
#             zip(data.metadata, outs, output_wh)
#         ):
#             track_ids, boxes, scores, class_ids, _ = out
#             # from vis4d.vis.image import imshow_bboxes
#             # import matplotlib.pyplot as plt
#             # img = imshow_bboxes(images[i], boxes, scores, class_ids, track_ids)
#             # import os
#             # os.makedirs(f"example/{metadata.videoName}/", exist_ok=True)
#             # plt.imsave(f"example/{metadata.videoName}/{str(metadata.frameIndex).zfill(5)}.jpg", img)
#             # postprocess for eval
#             dets = Boxes2D(
#                 torch.cat([boxes, scores.unsqueeze(-1)], -1),
#                 class_ids=class_ids,
#                 track_ids=track_ids,
#             )
#             dets.postprocess((metadata.size.width, metadata.size.height), wh)

#             prediction = copy.deepcopy(metadata)
#             prediction.labels = dets.to_scalabel(class_ids_to_name)
#             preds.append(prediction)
#             gts.append(copy.deepcopy(metadata))

#     _, log_str = val_loader.evaluate("track", preds, gts)
#     print(log_str)


# ## training loop
# def training_loop(model):
#     """Training loop."""
#     running_losses = {}
#     for epoch in range(num_epochs):
#         model.train()
#         for i, data in enumerate(train_loader):
#             data = InputSample.cat(data[0], device)

#             tic = perf_counter()
#             inputs, inputs_hw, gt_boxes, gt_class_ids = (
#                 data.images.tensor,
#                 [(wh[1], wh[0]) for wh in data.images.image_sizes],
#                 [x.boxes for x in data.targets.boxes2d],
#                 [x.class_ids for x in data.targets.boxes2d],
#             )

#             # zero the parameter gradients
#             optimizer.zero_grad()

#             # forward + backward + optimize
#             rpn_losses, rcnn_losses, outputs = model(
#                 normalize(inputs), inputs_hw, gt_boxes, gt_class_ids
#             )
#             total_loss = sum((*rpn_losses, *rcnn_losses))
#             total_loss.backward()
#             optimizer.step()
#             toc = perf_counter()

#             # print statistics
#             losses = dict(
#                 time=toc - tic,
#                 loss=total_loss,
#                 **rpn_losses._asdict(),
#                 **rcnn_losses._asdict(),
#             )
#             for k, v in losses.items():
#                 if k in running_losses:
#                     running_losses[k] += v
#                 else:
#                     running_losses[k] = v
#             if i % log_step == (log_step - 1):
#                 # model.visualize_proposals(inputs, outputs)
#                 log_str = f"[{epoch + 1}, {i + 1:5d} / {len(train_loader)}] "
#                 for k, v in running_losses.items():
#                     log_str += f"{k}: {v / log_step:.3f}, "
#                 print(log_str.rstrip(", "))
#                 running_losses = {}

#         scheduler.step()
#         torch.save(
#             model.state_dict(),
#             f"vis4d-workspace/qdtrack_epoch_{epoch + 1}.pt",
#         )
#         validation_loop(model)
#     print("training done.")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="qdtrack bdd train/eval.")
#     parser.add_argument(
#         "-c", "--ckpt", default=None, help="path of model to eval"
#     )
#     args = parser.parse_args()
#     if args.ckpt is None:
#         training_loop(qdtrack)
#     else:
#         if args.ckpt == "pretrained":
#             from mmcv.runner.checkpoint import load_checkpoint

#             weights = "./qdtrack_r50_65point7.ckpt"
#             load_checkpoint(qdtrack.backbone, weights, revise_keys=REV_KEYS)
#             load_checkpoint(qdtrack.fpn, weights, revise_keys=REV_KEYS)
#             load_checkpoint(
#                 qdtrack.faster_rcnn_heads, weights, revise_keys=REV_KEYS
#             )
#             load_checkpoint(qdtrack.qdtracker, weights, revise_keys=REV_KEYS)
#         else:
#             ckpt = torch.load(args.ckpt)
#             qdtrack.load_state_dict(ckpt)
#         validation_loop(qdtrack)
