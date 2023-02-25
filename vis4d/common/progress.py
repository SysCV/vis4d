"""This module contains utilities for progress bar."""
from __future__ import annotations

import datetime

import torch

from ..common.time import Timer


def compose_log_str(
    prefix: str,
    batch_idx: int,
    total_batches: int | float,
    timer: Timer,
    metrics: None | dict[str, int | float | torch.Tensor] = None,
) -> str:
    """Compose log str from given information."""
    time_sec_tot = timer.time()
    time_sec_avg = time_sec_tot / batch_idx
    eta_sec = time_sec_avg * (total_batches - batch_idx)
    if not eta_sec == float("inf"):
        eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
    else:  # pragma: no cover
        eta_str = "---"

    metrics_list: list[str] = []
    if metrics is not None:
        for k, v in metrics.items():
            name = k.split("/")[-1]  # remove prefix, e.g. train/loss
            if isinstance(v, (torch.Tensor, float)):
                if v < 1e-3:
                    kv_str = f"{name}: {v:.6f}"
                else:
                    kv_str = f"{name}: {v:.4f}"
            else:
                kv_str = f"{name}: {v}"
            if name == "loss":  # put total loss first
                metrics_list.insert(0, kv_str)
            else:
                metrics_list.append(kv_str)

    time_str = f"ETA: {eta_str}, " + (
        f"{time_sec_avg:.2f}s/it"
        if time_sec_avg > 1
        else f"{1/time_sec_avg:.2f}it/s"
    )
    logging_str = f"{prefix}: {batch_idx - 1}/{total_batches}, {time_str}"
    if len(metrics_list) > 0:
        logging_str += ", " + ", ".join(metrics_list)
    return logging_str
