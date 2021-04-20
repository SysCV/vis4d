"""Testcases for dataset build functions."""
import unittest

from detectron2.data import DatasetCatalog, MetadataCatalog

from openmt.data.build import get_detection_dataset_dicts


class TestBuild(unittest.TestCase):
    """build Testcase class."""

    def test_get_detection_dataset_dicts(self) -> None:
        """Testcase for get_detection_dataset_dicts."""
        names = ["a", "b"]
        cats = ["car"]
        data_dicts = [
            [dict(annotations=[dict(category_id=0)])],
            [dict(annotations=[dict(category_id=0)])],
        ]

        DatasetCatalog.register(names[0], lambda: data_dicts[0])
        DatasetCatalog.register(names[1], lambda: data_dicts[1])

        for name in names:
            meta = MetadataCatalog.get(name)
            meta.thing_classes = cats
            meta.idx_to_class_mapping = dict(enumerate(cats))

        dataset_dicts = get_detection_dataset_dicts(names, True, False)
        self.assertEqual(dataset_dicts, [d[0] for d in data_dicts])
