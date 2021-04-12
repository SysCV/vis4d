"""Test cases for openMT modeling utils."""
import unittest

from .utils import select_keyframe


class TestStructures(unittest.TestCase):
    """Test cases openMT modeling utils."""

    def test_keyframe_selection(self) -> None:
        """Testcase for keyframe selection."""
        possible_idcs = list(range(5))
        key, ref = select_keyframe(5, strategy="random")
        self.assertTrue(all(idx in possible_idcs for idx in [key] + ref))
        key, ref = select_keyframe(5, strategy="last")
        self.assertTrue(all(idx in possible_idcs for idx in [key] + ref))
        self.assertTrue(key == 4)
        key, ref = select_keyframe(5, strategy="first")
        self.assertTrue(all(idx in possible_idcs for idx in [key] + ref))
        self.assertTrue(key == 0)
