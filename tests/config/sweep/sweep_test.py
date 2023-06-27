"""Vis4d Sweep Config Tests."""


import unittest

import numpy as np
from ml_collections import ConfigDict

from vis4d.config.common.types import ExperimentConfig
from vis4d.config.replicator import replicate_config
from vis4d.config.util.sweep import grid_search


class TestSweep(unittest.TestCase):
    """Test for config sweeps."""

    def test_one_param_sweep(self) -> None:
        """Test if one param config sweep works"""

        expected = np.linspace(0.001, 0.01, 3)

        exp_names_expected = [f"test_lr_{lr:.3f}_" for lr in expected]

        sweep_config = grid_search("lr", list(expected))
        sweep_config.suffix = "lr_{lr:.3f}_"
        config = ExperimentConfig()
        config.lr = 0
        config.experiment_name = "test"
        config.value_mode()
        config_iter = replicate_config(
            config,
            method=sweep_config.method,
            sampling_args=sweep_config.sampling_args,
            fstring=sweep_config.get("suffix", ""),
        )

        lr_actual: list[float] = []
        exp_names: list[str] = []
        for c in config_iter:
            lr_actual.append(c.lr)
            exp_names.append(c.experiment_name)
        lr_actual = np.array(lr_actual)  # type: ignore

        self.assertTrue(np.allclose(expected, lr_actual))
        for pred, gt in zip(exp_names, exp_names_expected):
            self.assertEqual(pred, gt)

    def test_one_nested_param_sweep(self) -> None:
        """Test if one param config sweep works when it is nested,"""

        expected = np.linspace(0.001, 0.01, 3)

        exp_names_expected = [f"test_lr_{lr:.3f}_" for lr in expected]

        sweep_config = grid_search("params.lr", list(expected))
        sweep_config.suffix = "lr_{params.lr:.3f}_"
        config = ExperimentConfig()
        config.params = ConfigDict()
        config.params.lr = 0
        config.experiment_name = "test"
        config.value_mode()
        config_iter = replicate_config(
            config,
            method=sweep_config.method,
            sampling_args=sweep_config.sampling_args,
            fstring=sweep_config.get("suffix", ""),
        )

        lr_actual: list[float] = []
        exp_names: list[str] = []
        for c in config_iter:
            lr_actual.append(c.params.lr)
            exp_names.append(c.experiment_name)
        lr_actual = np.array(lr_actual)  # type: ignore

        self.assertTrue(np.allclose(expected, lr_actual))
        for pred, gt in zip(exp_names, exp_names_expected):
            self.assertEqual(pred, gt)

    def test_two_param_sweeps(self) -> None:
        """Test to sweep over two parameters (lr, bs)."""

        learning_rates = np.linspace(0.001, 0.01, 3)
        batch_sizes = [4, 8, 16]

        exp_names_expected = []
        lr_expected = []
        bs_expected = []
        for lr in learning_rates:
            for bs in batch_sizes:
                exp_names_expected.append(f"test_lr_{lr:.3f}_bs_{bs}_")
                lr_expected.append(lr)
                bs_expected.append(bs)

        sweep_config = grid_search(
            ["lr", "bs"], [list(learning_rates), batch_sizes]
        )
        sweep_config.suffix = "lr_{lr:.3f}_bs_{bs}_"
        config = ExperimentConfig()
        config.lr = 0
        config.experiment_name = "test"
        config.value_mode()
        config_iter = replicate_config(
            config,
            method=sweep_config.method,
            sampling_args=sweep_config.sampling_args,
            fstring=sweep_config.get("suffix", ""),
        )

        lr_actual: list[float] = []
        bs_actual: list[int] = []
        exp_names: list[str] = []
        for c in config_iter:
            lr_actual.append(c.lr)
            bs_actual.append(c.bs)
            exp_names.append(c.experiment_name)

        self.assertTrue(np.allclose(lr_expected, lr_actual))
        self.assertTrue(np.allclose(bs_expected, bs_actual))
        for pred, gt in zip(exp_names, exp_names_expected):
            self.assertEqual(pred, gt)
