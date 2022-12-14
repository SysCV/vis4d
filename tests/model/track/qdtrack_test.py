"""QDTrack model test file."""
import unittest

import torch
from mmcv.runner.checkpoint import load_checkpoint

from tests.util import get_test_file
from vis4d.data.const import CommonKeys
from vis4d.data.datasets.scalabel import Scalabel
from vis4d.data.loader import DataPipe, build_inference_dataloaders
from vis4d.data.transforms.normalize import normalize_image
from vis4d.data.transforms.pad import pad_image
from vis4d.model.track.qdtrack import FasterRCNNQDTrack

REV_KEYS = [
    (r"^detector.rpn_head.mm_dense_head\.", "rpn_head."),
    (r"\.rpn_reg\.", ".rpn_box."),
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
    (r"\.conv.weight", ".weight"),
    (r"\.conv.bias", ".bias"),
]


class QDTrackTest(unittest.TestCase):
    """QDTrack class tests."""

    model_weights = "https://dl.cv.ethz.ch/vis4d/qdtrack_bdd100k_frcnn_res50_heavy_augs.ckpt"

    def test_inference(self):
        """Inference test.

        Run::
            >>> pytest vis4d/op/track/qdtrack_test.py::QDTrackTest::test_inference
        """
        qdtrack = FasterRCNNQDTrack(num_classes=8)
        load_checkpoint(
            qdtrack,
            self.model_weights,
            map_location=torch.device("cpu"),
            revise_keys=REV_KEYS,
        )

        data_root = get_test_file(
            "track/bdd100k-samples/images", rel_path="run"
        )
        annotations = get_test_file(
            "track/bdd100k-samples/labels", rel_path="run"
        )
        config = get_test_file(
            "track/bdd100k-samples/config.toml", rel_path="run"
        )
        test_data = DataPipe(
            Scalabel(data_root, annotations, config_path=config),
            preprocess_fn=normalize_image(),
        )
        batch_fn = pad_image()
        batch_size = 2
        test_loader = build_inference_dataloaders(
            test_data,
            samples_per_gpu=batch_size,
            workers_per_gpu=0,
            batchprocess_fn=batch_fn,
        )[0]

        with torch.no_grad():
            for data in enumerate(test_loader):
                # assume: inputs are consecutive frames
                images = data[CommonKeys.images]
                inputs_hw = data[CommonKeys.input_hw]
                # TODO frame ids need to be implemented properly
                frame_ids = [CommonKeys.frame_ids]

                with torch.no_grad():
                    tracks = qdtrack(images, inputs_hw, frame_ids)

                # TODO test bdd100k val numbers and convert to results test

                # import numpy as np
                # from vis4d.vis.functional import imshow_bboxes
                # from vis4d.vis.util import preprocess_boxes, preprocess_image
                # for img, trk in zip(images, tracks):
                #     track_ids, boxes, scores, class_ids, _ = trk
                #     imshow_bboxes(
                #         np.array(preprocess_image(img)),
                #         boxes.numpy(),
                #         scores.numpy(),
                #         class_ids.numpy(),
                #         track_ids.numpy(),
                #     )

    def test_train(self):
        """Training test."""
        raise NotImplementedError

    def test_torchscript(self):
        """Test torchscipt export."""
        sample_images = torch.rand((2, 3, 512, 512))
        qdtrack = FasterRCNNQDTrack(num_classes=8)
        qdtrack_scripted = torch.jit.script(qdtrack)
        qdtrack_scripted(sample_images)
