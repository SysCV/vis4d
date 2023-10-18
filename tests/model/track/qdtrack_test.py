"""QDTrack model test file."""
import os.path as osp
import unittest

import torch

from tests.util import get_test_data, get_test_file
from vis4d.common.ckpt import load_model_checkpoint
from vis4d.data.const import CommonKeys as K
from vis4d.data.datasets.bdd100k import BDD100K
from vis4d.data.loader import DataPipe, build_inference_dataloaders
from vis4d.data.transforms.base import compose
from vis4d.data.transforms.normalize import NormalizeImages
from vis4d.data.transforms.pad import PadImages
from vis4d.data.transforms.resize import GenResizeParameters, ResizeImages
from vis4d.data.transforms.to_tensor import ToTensor
from vis4d.model.track.qdtrack import (
    REV_KEYS,
    YOLOX_REV_KEYS,
    FasterRCNNQDTrack,
    TrackOut,
    YOLOXQDTrack,
)
from vis4d.op.track.assignment import TrackIDCounter


class QDTrackTest(unittest.TestCase):
    """QDTrack class tests."""

    def test_inference_fasterrcnn(self):
        """Inference test for Faster R-CNN QDTrack.

        Run::
            >>> pytest tests/model/track/qdtrack_test.py::QDTrackTest::test_inference_fasterrcnn
        """
        TrackIDCounter.reset()  # reset track ID counter
        model_weights = (
            "https://dl.cv.ethz.ch/vis4d/"
            "qdtrack_bdd100k_frcnn_res50_heavy_augs.pt"
        )
        qdtrack = FasterRCNNQDTrack(num_classes=8)
        load_model_checkpoint(qdtrack, model_weights, rev_keys=REV_KEYS)
        qdtrack.eval()

        data_root = osp.join(get_test_data("bdd100k_test"), "track/images")
        annotations = osp.join(get_test_data("bdd100k_test"), "track/labels")
        config = osp.join(get_test_data("bdd100k_test"), "track/config.toml")
        preprocess_fn = compose(
            [
                GenResizeParameters((224, 384), keep_ratio=True),
                ResizeImages(),
                NormalizeImages(),
            ]
        )
        test_data = DataPipe(
            BDD100K(data_root, annotations, config_path=config),
            preprocess_fn=preprocess_fn,
        )
        batch_fn = compose([PadImages(), ToTensor()])
        batch_size = 2
        test_loader = build_inference_dataloaders(
            test_data,
            samples_per_gpu=batch_size,
            workers_per_gpu=1,
            batchprocess_fn=batch_fn,
        )[0]

        data = next(iter(test_loader))
        # assume: inputs are consecutive frames
        images = data[K.images]
        inputs_hw = data[K.input_hw]
        original_hw = data[K.original_hw]
        frame_ids = data[K.frame_ids]

        with torch.no_grad():
            tracks = qdtrack(images, inputs_hw, original_hw, frame_ids)
        assert isinstance(tracks, TrackOut)
        print("Testcase file:", get_test_file("qdtrack.pt"))
        testcase_gt = torch.load(get_test_file("qdtrack.pt"))
        for pred_entry, expected_entry in zip(tracks, testcase_gt):
            for pred, expected in zip(pred_entry, expected_entry):
                print("PREDICTION:", pred.shape, pred)
                print("EXPECTED:", expected.shape, expected)
                assert torch.isclose(pred, expected, atol=1e-4).all().item()

    def test_inference_yolox(self):
        """Inference test for YOLOX QDTrack."""
        TrackIDCounter.reset()  # reset track ID counter
        model_weights = (
            "https://dl.cv.ethz.ch/vis4d/qdtrack-yolox-ema_bdd100k.ckpt"
        )
        qdtrack = YOLOXQDTrack(num_classes=8)
        load_model_checkpoint(qdtrack, model_weights, rev_keys=YOLOX_REV_KEYS)
        qdtrack.eval()

        data_root = osp.join(get_test_data("bdd100k_test"), "track/images")
        annotations = osp.join(get_test_data("bdd100k_test"), "track/labels")
        config = osp.join(get_test_data("bdd100k_test"), "track/config.toml")
        preprocess_fn = compose(
            [
                GenResizeParameters(
                    (224, 384), keep_ratio=False, align_long_edge=True
                ),
                ResizeImages(),
            ]
        )
        test_data = DataPipe(
            BDD100K(data_root, annotations, config_path=config),
            preprocess_fn=preprocess_fn,
        )
        batch_fn = compose([PadImages(pad2square=True), ToTensor()])
        batch_size = 2
        test_loader = build_inference_dataloaders(
            test_data,
            samples_per_gpu=batch_size,
            workers_per_gpu=1,
            batchprocess_fn=batch_fn,
        )[0]

        data = next(iter(test_loader))
        # assume: inputs are consecutive frames
        images = data[K.images]
        inputs_hw = data[K.input_hw]
        original_hw = data[K.original_hw]
        frame_ids = data[K.frame_ids]

        with torch.no_grad():
            tracks = qdtrack(images, inputs_hw, original_hw, frame_ids)
        assert isinstance(tracks, TrackOut)
        print("Testcase file:", get_test_file("qdtrack-yolox.pt"))
        testcase_gt = torch.load(get_test_file("qdtrack-yolox.pt"))
        for pred_entry, expected_entry in zip(tracks, testcase_gt):
            for pred, expected in zip(pred_entry, expected_entry):
                print("PREDICTION:", pred.shape, pred)
                print("EXPECTED:", expected.shape, expected)
                assert torch.isclose(pred, expected, atol=1e-4).all().item()
