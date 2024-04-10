"""Test cases for logging callback."""

import unittest

from tests.util import MOCKLOSS, MockModel
from vis4d.engine.callbacks import LoggingCallback, TrainerState


class TestLoggingCallback(unittest.TestCase):
    """Test cases for callback functions."""

    def setUp(self) -> None:
        """Setup callback."""
        self.callback = LoggingCallback(refresh_rate=1)

        self.trainer_state = TrainerState(
            current_epoch=0,
            num_epochs=0,
            global_step=0,
            num_steps=0,
            train_dataloader=None,
            num_train_batches=1,
            test_dataloader=None,
            num_test_batches=[1],
        )

    def test_on_train_epoch_start(self) -> None:
        """Test on_train_epoch_start function."""
        self.callback.on_train_epoch_start(
            self.trainer_state, MockModel(0), MOCKLOSS
        )

    def test_on_train_batch_end(self) -> None:
        """Test on_train_batch_end function."""
        self.trainer_state["metrics"] = {"loss1": 0, "loss2": 1}

        self.callback.on_train_batch_end(
            self.trainer_state,
            MockModel(0),
            MOCKLOSS,
            outputs={},
            batch={},
            batch_idx=0,
        )

        self.trainer_state.pop("metrics")

    def test_on_test_epoch_start(self) -> None:
        """Test on_test_epoch_start function."""
        self.callback.on_test_epoch_start(self.trainer_state, MockModel(0))

    def test_on_test_batch_end(self) -> None:
        """Test on_test_batch_end function."""
        self.callback.on_test_batch_end(
            self.trainer_state,
            MockModel(0),
            outputs={},
            batch={},
            batch_idx=0,
        )
