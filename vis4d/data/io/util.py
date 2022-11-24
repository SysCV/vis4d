"""Data I/O Utilities."""
from __future__ import annotations

import sys


def str_decode(str_bytes: bytes, encoding: None | str = None) -> str:
    """Decode to string from bytes."""
    if encoding is None:
        encoding = sys.getdefaultencoding()
    return str_bytes.decode(encoding)
