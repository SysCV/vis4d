"""QDTrack model test file."""
import os.path as osp
import unittest

import torch

from tests.util import get_test_data, get_test_file
from vis4d.data.const import CommonKeys
from vis4d.data.datasets.scalabel import Scalabel
from vis4d.data.loader import DataPipe, build_inference_dataloaders
from vis4d.data.transforms.normalize import normalize_image
from vis4d.data.transforms.pad import pad_image
from vis4d.engine.ckpt import load_model_checkpoint
from vis4d.model.track.qdtrack import FasterRCNNQDTrack


class QDTrackTest(unittest.TestCase):
    """QDTrack class tests."""

    model_weights = (
        "https://dl.cv.ethz.ch/vis4d/qdtrack_bdd100k_frcnn_res50_heavy_augs.pt"
    )

    def test_inference(self):
        """Inference test.

        Run::
            >>> pytest tests/model/track/qdtrack_test.py::QDTrackTest::test_inference
        """
        qdtrack = FasterRCNNQDTrack(num_classes=8)
        load_model_checkpoint(qdtrack, self.model_weights)

        data_root = osp.join(get_test_data("bdd100k_test"), "track/images")
        annotations = osp.join(get_test_data("bdd100k_test"), "track/labels")
        config = osp.join(get_test_data("bdd100k_test"), "track/config.toml")
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

        data = next(iter(test_loader))
        # assume: inputs are consecutive frames
        images = data[CommonKeys.images]
        inputs_hw = data[CommonKeys.input_hw]
        frame_ids = data[CommonKeys.frame_ids]

        with torch.no_grad():
            tracks = qdtrack(  # pylint: disable=unused-variable
                images, inputs_hw, frame_ids
            )

        # TODO: Fix test
        # testcase_gt = torch.load(get_test_file("qdtrack.pt"))
        # for pred, expected in zip(tracks, testcase_gt):
        #     for pred_entry, expected_entry in zip(pred, expected):
        #         pass
        #         assert (
        #             torch.isclose(pred_entry, expected_entry, atol=1e-4)
        #             .all()
        #             .item()
        #         )

    # def test_train(self): # TODO: Fix test
    #     """Training test."""
    #     pass

    # def test_torchscript(self): # TODO: Fix test
    #     """Test torchscipt export."""
    #     sample_images = torch.rand((2, 3, 512, 512))
    #     qdtrack = FasterRCNNQDTrack(num_classes=8)
    #     qdtrack_scripted = torch.jit.script(qdtrack)
    #     qdtrack_scripted(sample_images)
