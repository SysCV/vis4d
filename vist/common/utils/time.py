"""Utilities for timing."""
from time import perf_counter
from typing import no_type_check


@no_type_check
def timeit(func):
    """Function to be used as decorator to time a function."""

    def timed(*args, **kwargs):
        tic = perf_counter()
        result = func(*args, **kwargs)
        toc = perf_counter()
        print("%r  %2.2f ms" % (func.__name__, (toc - tic) * 1000))
        return result

    return timed
