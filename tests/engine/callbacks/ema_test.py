"""Test cases for EMA callback."""
import unittest

import torch

from tests.util import MOCKLOSS, MockModel
from vis4d.engine.callbacks import EMACallback, TrainerState
from vis4d.model.adapter import ModelEMAAdapter


class TestEMACallback(unittest.TestCase):
    """Test cases for callback functions."""

    def setUp(self) -> None:
        """Setup callback."""
        self.callback = EMACallback()
        self.callback.setup()

        self.model = ModelEMAAdapter(MockModel(0))
        self.model.ema_model.linear.weight.fill_(0)
        self.model.ema_model.linear.bias.fill_(0)

        self.trainer_state = TrainerState(
            current_epoch=0,
            num_epochs=0,
            global_step=0,
            train_dataloader=None,
            num_train_batches=None,
            test_dataloader=None,
            num_test_batches=None,
        )

    def test_ema_callback(self) -> None:
        """Test EMA callback function."""
        self.callback.on_train_batch_end(
            self.trainer_state,
            self.model,
            MOCKLOSS,
            outputs={},
            batch={},
            batch_idx=0,
        )

        self.assertTrue(
            torch.isclose(
                self.model.ema_model.linear.weight,
                torch.tensor(
                    [
                        [
                            -3.7941e-06,
                            6.4374e-06,
                            1.7447e-05,
                            2.9962e-05,
                            7.2744e-06,
                            -1.1393e-05,
                            -3.0403e-05,
                            -4.3099e-05,
                            -4.1691e-06,
                            -1.0360e-06,
                        ]
                    ]
                ),
                atol=1e-4,
            ).all()
        )
        self.assertTrue(
            torch.isclose(
                self.model.ema_model.linear.bias,
                torch.tensor([[-2.4120e-05]]),
                atol=1e-4,
            ).all()
        )
