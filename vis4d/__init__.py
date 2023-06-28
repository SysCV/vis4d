"""Vis4D is a modular library for 4D scene understanding.

It contains common operators and models, data pipelines and training recipes
for a number of contemporary methods and provides a compositional framework
for further research and development of 4D Vision algorithms.
"""
import logging

from .engine.run import entrypoint

_root_logger = logging.getLogger()
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

# if root logger has handlers, propagate messages up and let root logger
# process them
if not _root_logger.hasHandlers():
    _logger.addHandler(logging.StreamHandler())
    _logger.propagate = False


if __name__ == "__main__":
    entrypoint()
