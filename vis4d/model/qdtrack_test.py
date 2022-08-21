"""QDTrack test file."""
import unittest

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from vis4d.common.datasets import bdd100k_track_map, bdd100k_track_sample
from vis4d.data.utils import transform_bbox
from vis4d.struct import Boxes2D


class SampleDataset(Dataset):
    def __init__(self):
        self.scalabel_data = bdd100k_track_sample()

    def __len__(self):
        return len(self.scalabel_data.frames)

    def __getitem__(self, item):
        frame = self.scalabel_data.frames[item]
        img = url_to_tensor(frame.url, size=(512, 512))
        labels = Boxes2D.from_scalabel(frame.labels, bdd100k_track_map)
        trans_mat = torch.eye(3)
        trans_mat[0, 0] = 512 / 1280
        trans_mat[1, 1] = 512 / 720
        labels.boxes[:, :4] = transform_bbox(trans_mat, labels.boxes[:, :4])
        return img, labels.boxes, labels.class_ids


def identity_collate(batch):
    return tuple(zip(*batch))


class QDTrackTest(unittest.TestCase):
    def test_inference(self):
        qdtrack = QDTrack()
        qdtrack.eval()
        with torch.no_grad():
            outs = qdtrack(sample_images)

    def test_train(self):
        qdtrack = QDTrack()

        optimizer = optim.SGD(qdtrack.parameters(), lr=0.001, momentum=0.9)

        train_data = SampleDataset()
        train_loader = DataLoader(
            train_data, batch_size=2, shuffle=True, collate_fn=identity_collate
        )

        running_losses = {}
        qdtrack.train()
        log_step = 1
        for epoch in range(2):
            for i, data in enumerate(train_loader):
                inputs, gt_boxes, gt_class_ids = data
                inputs = torch.cat(inputs)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = qdtrack(inputs, gt_boxes, gt_class_ids)
                total_loss = sum((*rpn_losses, *rcnn_losses))
                total_loss.backward()
                optimizer.step()

                # print statistics
                losses = dict(
                    loss=total_loss,
                    **rpn_losses._asdict(),
                    **rcnn_losses._asdict(),
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
        sample_images = torch.rand((2, 3, 512, 512))
        qdtrack = QDTrack()
        qdtrack_scripted = torch.jit.script(qdtrack)
        qdtrack_scripted(sample_images)


if proposal_sampler is not None:
    self.sampler = proposal_sampler
else:
    self.sampler = CombinedSampler(
        batch_size=256,
        positive_fraction=0.5,
        pos_strategy="instance_balanced",
        neg_strategy="iou_balanced",
    )

if proposal_matcher is not None:
    self.matcher = proposal_matcher
else:
    self.matcher = MaxIoUMatcher(
        thresholds=[0.3, 0.7],
        labels=[0, -1, 1],
        allow_low_quality_matches=False,
    )

# TODO will be part of training loop
sampling_results, sampled_boxes, sampled_targets = [], [], []
for i, (box, tgt) in enumerate(zip(boxes, targets)):
    sampling_result = match_and_sample_proposals(
        self.matcher,
        self.sampler,
        box,
        tgt.boxes2d,
        self.proposal_append_gt,
    )
    sampling_results.append(sampling_result)

    sampled_box = sampling_result.sampled_boxes
    sampled_tgt = sampling_result.sampled_targets
    positives = [l == 1 for l in sampling_result.sampled_labels]
    if i == 0:  # take only positives for keyframe (assumed at i=0)
        sampled_box = [b[p] for b, p in zip(sampled_box, positives)]
        sampled_tgt = [t[p] for t, p in zip(sampled_tgt, positives)]
    else:  # set track_ids to -1 for all negatives
        for pos, samp_tgt in zip(positives, sampled_tgt):
            samp_tgt.track_ids[~pos] = -1

    sampled_boxes.append(sampled_box)
    sampled_targets.append(sampled_tgt)
