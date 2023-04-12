"""QDTrack model test file."""
import os.path as osp
import unittest

import torch

from tests.util import get_test_data, get_test_file
from vis4d.data.const import CommonKeys
from vis4d.data.datasets.bdd100k import BDD100K
from vis4d.data.loader import DataPipe, build_inference_dataloaders
from vis4d.data.transforms.base import compose, compose_batch
from vis4d.data.transforms.normalize import NormalizeImage
from vis4d.data.transforms.pad import PadImages
from vis4d.data.transforms.to_tensor import ToTensor
from vis4d.engine.ckpt import load_model_checkpoint
from vis4d.model.track.qdtrack import FasterRCNNQDTrack, TrackOut, REV_KEYS


class QDTrackTest(unittest.TestCase):
    """QDTrack class tests."""

    # TODO: Fix test with reproduced design
    model_weights = (
        "https://dl.cv.ethz.ch/vis4d/qdtrack_bdd100k_frcnn_res50_heavy_augs.pt"
    )

    def test_inference(self):
        """Inference test.

        Run::
            >>> pytest tests/model/track/qdtrack_test.py::QDTrackTest::test_inference
        """
        qdtrack = FasterRCNNQDTrack(num_classes=8)
        load_model_checkpoint(qdtrack, self.model_weights, rev_keys=REV_KEYS)
        qdtrack.eval()

        data_root = osp.join(get_test_data("bdd100k_test"), "track/images")
        annotations = osp.join(get_test_data("bdd100k_test"), "track/labels")
        config = osp.join(get_test_data("bdd100k_test"), "track/config.toml")
        test_data = DataPipe(
            BDD100K(data_root, annotations, config_path=config),
            preprocess_fn=compose([NormalizeImage()]),
        )
        batch_fn = compose_batch([PadImages(), ToTensor()])
        batch_size = 2
        test_loader = build_inference_dataloaders(
            test_data,
            samples_per_gpu=batch_size,
            workers_per_gpu=0,
            batchprocess_fn=batch_fn,
        )[0]

        data = next(iter(test_loader))
        # assume: inputs are consecutive frames
        images = data[CommonKeys.images]
        inputs_hw = data[CommonKeys.input_hw]
        frame_ids = data[CommonKeys.frame_ids]

        with torch.no_grad():
            tracks = qdtrack(  # pylint: disable=unused-variable
                images, inputs_hw, frame_ids
            )
        assert isinstance(tracks, TrackOut)
        print("Testcase file:", get_test_file("qdtrack.pt"))
        testcase_gt = torch.load(get_test_file("qdtrack.pt"))
        for pred_entry, expected_entry in zip(tracks, testcase_gt):
            for pred, expected in zip(pred_entry, expected_entry):
                print("PREDICTION:", pred.shape, pred)
                print("EXPECTED:", expected.shape, expected)
                assert torch.isclose(pred, expected, atol=1e-4).all().item()

    # def test_train(self): # TODO: Fix test
    #     """Training test."""
    #     pass

    # def test_torchscript(self): # TODO: Fix test
    #     """Test torchscipt export."""
    #     sample_images = torch.rand((2, 3, 512, 512))
    #     qdtrack = FasterRCNNQDTrack(num_classes=8)
    #     qdtrack_scripted = torch.jit.script(qdtrack)
    #     qdtrack_scripted(sample_images)
