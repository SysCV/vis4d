"""Vis4D is a batteries-included 4D Computer Vision package.

It contains common operators and models, data pipelines and training recipes
for a number of contemporary methods and provides a compositional framework
for further research and development of 4D Vision algorithms.
"""
import logging

_root_logger = logging.getLogger()
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

# if root logger has handlers, propagate messages up and let root logger
# process them
if not _root_logger.hasHandlers():
    _logger.addHandler(logging.StreamHandler())
    _logger.propagate = False
