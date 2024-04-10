"""This module contains utilities for progress bar."""

from __future__ import annotations

import datetime

from torch import Tensor

from .time import Timer
from .typing import MetricLogs


def compose_log_str(
    prefix: str,
    cur_iter: int,
    total_iters: int,
    timer: Timer,
    metrics: None | MetricLogs = None,
) -> str:
    """Compose log str from given information."""
    time_sec_tot = timer.time()
    time_sec_avg = time_sec_tot / cur_iter
    eta_sec = time_sec_avg * (total_iters - cur_iter)
    if not eta_sec == float("inf"):
        eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
    else:  # pragma: no cover
        eta_str = "---"

    metrics_list: list[str] = []
    if metrics is not None:
        for k, v in metrics.items():
            name = k.split("/")[-1]  # remove prefix, e.g. train/loss
            if isinstance(v, (Tensor, float)):
                # display more digits for small values
                if abs(v) < 1e-3:  # type: ignore[operator]
                    kv_str = f"{name}: {v:.3e}"
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
    logging_str = f"{prefix}: {cur_iter}/{total_iters}, {time_str}"
    if len(metrics_list) > 0:
        logging_str += ", " + ", ".join(metrics_list)
    return logging_str
