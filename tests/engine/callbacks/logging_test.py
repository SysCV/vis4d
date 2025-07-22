"""Test cases for logging callback."""

import unittest

import lightning.pytorch as pl

from vis4d.engine.callbacks import LoggingCallback


class TestLoggingCallback(unittest.TestCase):
    """Test cases for callback functions."""

    def setUp(self) -> None:
        """Setup callback."""
        self.callback = LoggingCallback(refresh_rate=1)

        self.trainer = pl.Trainer()
        self.training_module = pl.LightningModule()

    def test_on_train_epoch_start(self) -> None:
        """Test on_train_epoch_start function."""
        self.callback.on_train_epoch_start(self.trainer, self.training_module)

    def test_on_train_batch_end(self) -> None:
        """Test on_train_batch_end function."""
        self.callback.on_train_batch_end(
            self.trainer,
            self.training_module,
            outputs={},
            batch={},
            batch_idx=0,
        )

    def test_on_test_epoch_start(self) -> None:
        """Test on_test_epoch_start function."""
        self.callback.on_test_epoch_start(self.trainer, self.training_module)

    def test_on_test_batch_end(self) -> None:
        """Test on_test_batch_end function."""
        self.callback.on_test_batch_end(
            self.trainer,
            self.training_module,
            outputs={},
            batch={},
            batch_idx=0,
        )
