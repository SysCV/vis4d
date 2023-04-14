"""Test case for config initialization."""

import unittest

from vis4d.config.util import class_config, instantiate_classes
from vis4d.data.transforms.resize import GenResizeParameters, ResizeBoxes2D


class TestConfigInstantiation(unittest.TestCase):
    """Test for config instantiation."""

    def test_instantiate_transforms(self) -> None:
        """Test if instantiation of a transform works."""
        conf = class_config(ResizeBoxes2D)
        instance = instantiate_classes(conf)
        self.assertTrue(isinstance(instance, ResizeBoxes2D))

    def test_instantiate_transforms_with_param(self) -> None:
        """Test if instantiation of a transform with parameters works."""
        conf = class_config(GenResizeParameters, shape=(10, 10))
        instance = instantiate_classes(conf)
        self.assertTrue(isinstance(instance, GenResizeParameters))
        self.assertTrue(instance.shape == (10, 10))
