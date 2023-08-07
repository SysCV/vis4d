"""Test cases for checkpoint callback."""
import shutil
import tempfile
import unittest

from torch.optim import SGD

from tests.util import MOCKLOSS, MockModel
from vis4d.config import class_config
from vis4d.engine.callbacks import CheckpointCallback, TrainerState

from ..optim.optimizer_test import get_optimizer


class TestCheckpointCallback(unittest.TestCase):
    """Test cases for callback functions."""

    def setUp(self) -> None:
        """Creates a tmp directory and setup callback."""
        self.test_dir = tempfile.mkdtemp()

        self.callback = CheckpointCallback(save_prefix=self.test_dir)

        self.callback.setup()

        optimizers, lr_scheulders = get_optimizer(
            MockModel(0), class_config(SGD, lr=0.01)
        )

        self.trainer_state = TrainerState(
            current_epoch=0,
            num_epochs=0,
            global_step=0,
            train_dataloader=None,
            num_train_batches=None,
            test_dataloader=None,
            num_test_batches=None,
            optimizers=optimizers,
            lr_schedulers=lr_scheulders,
        )

    def tearDown(self) -> None:
        """Removes the tmp directory after the test."""
        shutil.rmtree(self.test_dir)

    def test_on_train_epoch_end(self) -> None:
        """Test on_train_epoch_end function."""
        self.callback.on_train_epoch_end(
            self.trainer_state, MockModel(0), MOCKLOSS
        )
