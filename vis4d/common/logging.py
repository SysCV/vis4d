"""Vis4D logging unctions."""
from __future__ import annotations

import logging
import warnings

from .distributed import rank_zero_only
from .typing import ArgsType

log = logging.getLogger(__name__)


def _debug(*args: ArgsType, stacklevel: int = 2, **kwargs: ArgsType) -> None:
    kwargs["stacklevel"] = stacklevel
    log.debug(*args, **kwargs)


@rank_zero_only
def rank_zero_debug(
    *args: ArgsType, stacklevel: int = 4, **kwargs: ArgsType
) -> None:
    """Function used to log debug-level messages only on global rank 0."""
    _debug(*args, stacklevel=stacklevel, **kwargs)


def _info(*args: ArgsType, stacklevel: int = 2, **kwargs: ArgsType) -> None:
    kwargs["stacklevel"] = stacklevel
    log.info(*args, **kwargs)


@rank_zero_only
def rank_zero_info(
    *args: ArgsType, stacklevel: int = 4, **kwargs: ArgsType
) -> None:
    """Function used to log info-level messages only on global rank 0."""
    _info(*args, stacklevel=stacklevel, **kwargs)


def _warn(
    message: str | Warning, stacklevel: int = 2, **kwargs: ArgsType
) -> None:
    warnings.warn(message, stacklevel=stacklevel, **kwargs)


@rank_zero_only
def rank_zero_warn(
    message: str | Warning, stacklevel: int = 4, **kwargs: ArgsType
) -> None:
    """Function used to log warn-level messages only on global rank 0."""
    _warn(message, stacklevel=stacklevel, **kwargs)
