"""Test cases for checkpoint callback."""
import tempfile
import unittest

from tests.util import MockModel
from vis4d.engine.callbacks import CheckpointCallback, TrainerState
from vis4d.engine.connectors import DataConnector


class TestCheckpointCallback(unittest.TestCase):
    """Test cases for callback functions."""

    callback = CheckpointCallback(
        save_prefix=tempfile.mkdtemp(),
    )

    trainer_state = TrainerState(
        current_epoch=0,
        num_epochs=0,
        global_step=0,
        data_connector=DataConnector(),
    )

    def test_setup(self):
        """Test setup function."""
        self.callback.setup()

    def test_on_train_epoch_end(self) -> None:
        """Test on_train_epoch_end function."""
        self.callback.on_train_epoch_end(
            self.trainer_state,
            MockModel(0),
        )
