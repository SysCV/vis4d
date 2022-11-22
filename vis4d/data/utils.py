"""data utils."""
from __future__ import annotations

import sys
from time import perf_counter
from typing import no_type_check


def str_decode(str_bytes: bytes, encoding: None | str = None) -> str:
    """Decode to string from bytes."""
    if encoding is None:
        encoding = sys.getdefaultencoding()
    return str_bytes.decode(encoding)


@no_type_check
def timeit(func):
    """Function to be used as decorator to time a function."""

    def timed(*args, **kwargs):
        tic = perf_counter()
        result = func(*args, **kwargs)
        toc = perf_counter()
        print(f"{func.__name__}  {(toc - tic) * 1000:.2f} ms")
        return result

    return timed
