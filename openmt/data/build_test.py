"""Testcases for dataset build functions."""
import unittest

from detectron2.data import DatasetCatalog, MetadataCatalog
from scalabel.label.typing import Frame

from openmt.data.build import get_tracking_dataset_dicts


class TestBuild(unittest.TestCase):
    """build Testcase class."""

    def test_get_tracking_dataset_dicts(self) -> None:
        """Testcase for get_detection_dataset_dicts."""
        names = ["a", "b"]
        cats = ["car"]
        data_dicts = [
            [Frame(**dict(name="1", labels=[dict(id="1", category=0)]))],
            [Frame(**dict(name="1", labels=[dict(id="1", category=0)]))],
        ]

        DatasetCatalog.register(names[0], lambda: data_dicts[0])
        DatasetCatalog.register(names[1], lambda: data_dicts[1])

        for name in names:
            meta = MetadataCatalog.get(name)
            meta.thing_classes = cats
            meta.idx_to_class_mapping = dict(enumerate(cats))

        dataset_dicts = get_tracking_dataset_dicts(names, True, False)
        self.assertEqual(dataset_dicts, [d[0] for d in data_dicts])

        names = ["c", "d", "e"]
        data_dicts = [
            [Frame(**dict(name="1", labels=[]))],
            [Frame(**dict(name="1", labels=None))],
            [
                Frame(
                    **dict(
                        name="1",
                        labels=[
                            dict(
                                id="1",
                                category=0,
                                attributes=dict(crowd=False),
                            )
                        ],
                    )
                )
            ],
        ]

        DatasetCatalog.register(names[0], lambda: data_dicts[0])
        DatasetCatalog.register(names[1], lambda: data_dicts[1])
        DatasetCatalog.register(names[2], lambda: data_dicts[2])

        for name in names:
            meta = MetadataCatalog.get(name)
            meta.thing_classes = cats
            meta.idx_to_class_mapping = dict(enumerate(cats))

        dataset_dicts = get_tracking_dataset_dicts(names, True, False)
        self.assertEqual(len(dataset_dicts), 1)
