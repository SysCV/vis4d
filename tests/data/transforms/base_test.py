"""Test base transforms."""
from __future__ import annotations

from vis4d.data.transforms.base import (
    BatchRandomApply,
    BatchTransform,
    RandomApply,
    Transform,
)


@Transform("test_data", "test_data")
class TransformTest:
    """Test transform."""

    def __call__(self, test_data: int):
        """Call."""
        return test_data + 1


@BatchTransform("test_data", "test_data")
class BatchTransformTest:
    """Test batch transform."""

    def __call__(self, batch: list[int]):
        """Call."""
        return [x + 1 for x in batch]


def test_random_apply():
    """Test random apply."""
    random_apply = RandomApply([TransformTest()], 1.0)
    batch_random_apply = BatchRandomApply([BatchTransformTest()], 1.0)

    test_data = {"test_data": 0}
    batch = [{"test_data": 0}, {"test_data": 1}]

    assert random_apply(test_data)["test_data"] == 1
    assert batch_random_apply(batch)[0]["test_data"] == 1
    assert batch_random_apply(batch)[1]["test_data"] == 3

    random_apply = RandomApply([TransformTest()], probability=0.0)
    batch_random_apply = BatchRandomApply(
        [BatchTransformTest()], probability=0.0
    )

    test_data = {"test_data": 0}
    batch = [{"test_data": 0}, {"test_data": 1}]

    assert random_apply(test_data)["test_data"] == 0
    assert batch_random_apply(batch)[0]["test_data"] == 0
    assert batch_random_apply(batch)[1]["test_data"] == 1
