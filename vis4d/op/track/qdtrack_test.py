"""QDTrack test file."""
import unittest
from typing import List, Tuple, Union

import torch
from mmcv.runner.checkpoint import load_checkpoint
from torch import optim
from torch.utils.data import DataLoader

from vis4d.op.box.samplers import match_and_sample_proposals
from vis4d.op.detect.faster_rcnn_test import (
    FPN,
    FasterRCNNHead,
    ResNet,
    SampleDataset,
    get_default_anchor_generator,
    get_default_rcnn_box_encoder,
    get_default_rpn_box_encoder,
    identity_collate,
)
from vis4d.op.detect.rcnn import RCNNLoss, RCNNLosses, RoI2Det
from vis4d.op.detect.rpn import RPNLoss, RPNLosses
from vis4d.op.track.qdtrack import (
    QDSimilarityHead,
    QDTrackAssociation,
    QDTrackInstanceSimilarityLoss,
    QDTrackInstanceSimilarityLosses,
    get_default_box_matcher,
    get_default_box_sampler,
)
from vis4d.state.track.qdtrack import QDTrackMemory, QDTrackState


def pad(images: torch.Tensor, stride=32) -> torch.Tensor:
    """Pad image tensor to be compatible with stride."""
    N, C, H, W = images.shape
    pad = lambda x: (x + (stride - 1)) // stride * stride
    pad_hw = pad(H), pad(W)
    padded_imgs = images.new_zeros((N, C, *pad_hw))
    padded_imgs[:, :, :H, :W] = images
    return padded_imgs


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
    (r"^similarity_head\.", ""),
    ("\.conv.weight", ".weight"),
    ("\.conv.bias", ".bias"),
]


def split_key_ref(
    entries: Union[List[torch.Tensor], torch.Tensor], num_ref_views: int = 1
):
    """Split entries into key and reference views."""
    batch_size = len(entries)
    key_entries = [entries[i] for i in range(0, batch_size, num_ref_views + 1)]
    ref_entries = [
        entries[i + 1 : i + num_ref_views]
        for i in range(0, batch_size, num_ref_views + 1)
    ]
    return key_entries, ref_entries


@torch.no_grad()
def sample_proposals(
    box_matcher,
    box_sampler,
    boxes: List[torch.Tensor],
    target_boxes: List[torch.Tensor],
    target_track_ids: List[torch.Tensor],
    keyframe: bool = False,
    proposal_append_gt: bool = True,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Sample proposals for instance similarity learning."""
    if proposal_append_gt:
        boxes = [torch.cat([b, t]) for b, t in zip(boxes, target_boxes)]

    (
        sampled_box_indices,
        sampled_target_indices,
        sampled_labels,
    ) = match_and_sample_proposals(
        box_matcher, box_sampler, boxes, target_boxes
    )

    sampled_boxes, sampled_track_ids = [], []
    for boxs, track_ids, box_indices, target_indices, labels in zip(
        boxes,
        target_track_ids,
        sampled_box_indices,
        sampled_target_indices,
        sampled_labels,
    ):
        positives = labels == 1
        if keyframe:
            sampled_boxes.append(boxs[box_indices][positives])
            sampled_track_ids.append(track_ids[target_indices[positives]])
        else:  # set track_ids to -1 for all negatives
            sampled_boxes.append(boxs[box_indices])
            samp_track_ids = track_ids[target_indices]
            samp_track_ids[~positives] = -1
            sampled_track_ids.append(samp_track_ids)

    return sampled_boxes, sampled_track_ids


def key_ref_collate(batch):
    """Collate as key, ref pair."""
    key_batch, ref_batch = [], []
    for batch_elem in batch:
        key_data, ref_data = batch_elem
        key_batch.append(key_data)
        ref_batch.append(ref_data)
    return tuple(zip(*key_batch)), tuple(zip(*ref_batch))


class QDTrackTest(unittest.TestCase):
    """QDTrack class tests."""

    def test_inference(self):
        """Inference test.

        Run::
            >>> pytest vis4d/op/track/qdtrack_test.py::QDTrackTest::test_inference
        """  # pylint: disable=line-too-long # Disable the line length requirement becase of the cmd line prompts
        base = ResNet("resnet50")
        fpn = FPN(base.out_channels[2:], 256)
        faster_rcnn = FasterRCNNHead(num_classes=8)
        transform_detections = RoI2Det(
            faster_rcnn.rcnn_box_encoder, score_threshold=0.05
        )
        similarity_head = QDSimilarityHead()
        track_memory = QDTrackMemory(memory_limit=10)
        associate = QDTrackAssociation()

        load_checkpoint(
            base,
            "./qdtrack_r50_65point7.ckpt",
            map_location=torch.device("cpu"),
            revise_keys=REV_KEYS,
        )

        load_checkpoint(
            fpn,
            "./qdtrack_r50_65point7.ckpt",
            map_location=torch.device("cpu"),
            revise_keys=REV_KEYS,
        )

        load_checkpoint(
            faster_rcnn,
            "./qdtrack_r50_65point7.ckpt",
            map_location=torch.device("cpu"),
            revise_keys=REV_KEYS,
        )

        load_checkpoint(
            similarity_head,
            "./qdtrack_r50_65point7.ckpt",
            map_location=torch.device("cpu"),
            revise_keys=REV_KEYS,
        )

        test_data = SampleDataset()

        batch_size = 2
        test_loader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=identity_collate,
        )

        with torch.no_grad():
            for data in test_loader:
                # assume: inputs are consecutive frames
                inputs, inputs_hw, _, _, _, frame_ids = data
                images = pad(torch.cat(inputs))

                features = base(images)
                features = fpn(features)

                detector_out = faster_rcnn(features, inputs_hw)
                boxes, scores, class_ids = transform_detections(
                    *detector_out.roi, detector_out.proposals.boxes, inputs_hw
                )

                embeddings = similarity_head(features, boxes)

                tracks = []
                for frame_id, box, score, cls_id, embeds in zip(
                    frame_ids, boxes, scores, class_ids, embeddings
                ):
                    # reset graph at begin of sequence
                    if frame_id == 0:
                        track_memory.reset()

                    cur_memory = track_memory.get_current_tracks(box.device)
                    track_ids, filter_indices = associate(
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
                    track_memory.update(data)
                    tracks.append(track_memory.last_frame)

                from vis4d.vis.image import imshow_bboxes

                for img, boxs, score, cls_id in zip(
                    images, boxes, scores, class_ids
                ):
                    imshow_bboxes(img, boxs, score, cls_id)

                for img, trk in zip(images, tracks):
                    track_ids, boxes, scores, class_ids, _ = trk
                    imshow_bboxes(img, boxes, scores, class_ids, track_ids)
                # TODO test bdd100k val numbers and convert to results comparison

    def test_train(self):
        """Training test."""
        anchor_gen = get_default_anchor_generator()
        rpn_bbox_encoder = get_default_rpn_box_encoder()
        rcnn_bbox_encoder = get_default_rcnn_box_encoder()
        base = ResNet("resnet50", pretrained=True, trainable_layers=3)
        fpn = FPN(base.out_channels[2:], 256)
        faster_rcnn = FasterRCNNHead(
            num_classes=8,
            anchor_generator=anchor_gen,
            rpn_box_encoder=rpn_bbox_encoder,
            rcnn_box_encoder=rcnn_bbox_encoder,
        )
        similarity_head = QDSimilarityHead()
        box_matcher = get_default_box_matcher()
        box_sampler = get_default_box_sampler()
        rpn_loss = RPNLoss(anchor_gen, rpn_bbox_encoder)
        rcnn_loss = RCNNLoss(rcnn_bbox_encoder, num_classes=8)
        qdtrack_loss = QDTrackInstanceSimilarityLoss()

        optimizer = optim.SGD(
            [
                *base.parameters(),
                *fpn.parameters(),
                *faster_rcnn.parameters(),
                *similarity_head.parameters(),
            ],
            lr=0.001,
            momentum=0.9,
        )

        train_data = SampleDataset(sample_reference_view=True)
        train_loader = DataLoader(
            train_data, batch_size=2, shuffle=True, collate_fn=key_ref_collate
        )

        def train_step(
            data,
        ) -> Tuple[RPNLosses, RCNNLosses, QDTrackInstanceSimilarityLosses]:
            """Train step implementation."""
            key_data, ref_data = data
            (
                key_inputs,
                key_hw,
                key_gt_boxes,
                key_gt_class_ids,
                key_gt_track_ids,
                _,
            ) = key_data
            key_inputs = torch.cat(key_inputs)

            ref_inputs, ref_hw, ref_gt_boxes, _, ref_gt_track_ids, _ = ref_data
            ref_inputs = torch.cat(ref_inputs)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            features = base(torch.cat([key_inputs, ref_inputs]))
            features = fpn(features)

            key_features = [f[: len(key_inputs)] for f in features]
            ref_features = [f[len(key_inputs) :] for f in features]

            # keyframe detection
            key_detector_out = faster_rcnn(
                key_features, key_hw, key_gt_boxes, key_gt_class_ids
            )

            # detector losses only on keyframes
            rpn_losses = rpn_loss(
                key_detector_out.rpn.cls,
                key_detector_out.rpn.box,
                key_gt_boxes,
                key_hw,
            )
            rcnn_losses = rcnn_loss(
                key_detector_out.roi.cls_score,
                key_detector_out.roi.bbox_pred,
                key_detector_out.sampled_proposals.boxes,
                key_detector_out.sampled_targets.labels,
                key_detector_out.sampled_targets.boxes,
                key_detector_out.sampled_targets.classes,
            )

            # keyframe embedding extraction
            sampled_key_boxes, sampled_key_track_ids = sample_proposals(
                box_matcher,
                box_sampler,
                key_detector_out.proposals.boxes,
                key_gt_boxes,
                key_gt_track_ids,
            )
            key_embeddings = similarity_head(key_features, sampled_key_boxes)

            # reference frame detection, embedding extraction
            with torch.no_grad():
                ref_detector_out = faster_rcnn(ref_features, ref_hw)

            sampled_ref_boxes, sampled_ref_track_ids = sample_proposals(
                box_matcher,
                box_sampler,
                ref_detector_out.proposals.boxes,
                ref_gt_boxes,
                ref_gt_track_ids,
            )

            ref_embeddings = similarity_head(ref_features, sampled_ref_boxes)

            track_losses = qdtrack_loss(
                key_embeddings,
                [ref_embeddings],
                sampled_key_track_ids,
                [sampled_ref_track_ids],
            )
            return rpn_losses, rcnn_losses, track_losses

        running_losses = {}
        log_step = 1
        for epoch in range(2):
            for i, data in enumerate(train_loader):
                rpn_losses, rcnn_losses, track_losses = train_step(data)
                total_loss = sum((*rpn_losses, *rcnn_losses, *track_losses))
                total_loss.backward()
                optimizer.step()

                # print statistics
                losses = dict(
                    loss=total_loss,
                    **rpn_losses._asdict(),
                    **rcnn_losses._asdict(),
                    **track_losses._asdict(),
                )
                for k, v in losses.items():
                    if k in running_losses:
                        running_losses[k] += v
                    else:
                        running_losses[k] = v
                if i % log_step == (log_step - 1):
                    log_str = f"[{epoch + 1}, {i + 1:5d}] "
                    for k, v in running_losses.items():
                        log_str += f"{k}: {v / log_step:.3f}, "
                    print(log_str.rstrip(", "))
                    running_losses = {}

    def test_torchscript(self):
        """Test torchscipt export."""
        sample_images = torch.rand((2, 3, 512, 512))
        qdtrack = QDTrack()
        qdtrack_scripted = torch.jit.script(qdtrack)
        qdtrack_scripted(sample_images)
