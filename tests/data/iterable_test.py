"""Iterable datasets tests."""

from __future__ import annotations

import numpy as np
from torch import Tensor

from tests.util import get_test_data
from vis4d.data.const import CommonKeys as K
from vis4d.data.data_pipe import DataPipe
from vis4d.data.datasets import S3DIS
from vis4d.data.iterable import SubdividingIterableDataset
from vis4d.data.loader import (
    build_inference_dataloaders,
    build_train_dataloader,
)
from vis4d.data.transforms import compose
from vis4d.data.transforms.point_sampling import (
    GenerateBlockSamplingIndices,
    GenFullCovBlockSamplingIndices,
    SampleColors,
    SampleInstances,
    SamplePoints,
    SampleSemantics,
)


def test_train_loader_3d() -> None:
    """Test the data loading pipeline for 3D Data."""
    s3dis = S3DIS(data_root=get_test_data("s3d_test"))
    batch_size = 2
    preprocess_fn = compose(
        [
            GenerateBlockSamplingIndices(
                num_pts=1024, block_dimensions=(1, 1, 4)
            ),
            SampleInstances(),
            SampleSemantics(),
            SampleColors(),
            SamplePoints(),
        ]
    )

    datapipe = DataPipe(s3dis, preprocess_fn)
    train_loader = build_train_dataloader(
        datapipe,
        samples_per_gpu=batch_size,
        collate_keys=[
            K.points3d,
            K.colors3d,
            K.instances3d,
            K.semantics3d,
        ],
    )

    sample = next(iter(train_loader))
    assert isinstance(sample[K.colors3d], Tensor)
    assert isinstance(sample[K.points3d], Tensor)
    assert isinstance(sample[K.semantics3d], Tensor)
    assert isinstance(sample[K.instances3d], Tensor)
    assert batch_size == sample[K.colors3d].shape[0]
    assert batch_size == sample[K.points3d].shape[0]
    assert batch_size == sample[K.semantics3d].shape[0]

    assert sample[K.semantics3d].shape[1] == 1024
    assert sample[K.points3d].shape[1] == 1024
    assert sample[K.colors3d].shape[1] == 1024
    assert sample[K.instances3d].shape[1] == 1024


def test_loader_3d_batched() -> None:
    """Test the data loading pipeline for 3D Data with full scene sampling."""
    s3dis = S3DIS(data_root=get_test_data("s3d_test"))
    keys = (K.points3d, K.colors3d, K.instances3d, K.semantics3d)
    batch_size = 2
    preprocess_fn = compose(
        [
            GenFullCovBlockSamplingIndices(
                num_pts=1024, block_dimensions=(1, 1, 4)
            ),
            SampleInstances(),
            SampleSemantics(),
            SampleColors(),
            SamplePoints(),
        ]
    )

    datapipe = DataPipe(s3dis, preprocess_fn)
    inference_loader = build_inference_dataloaders(
        SubdividingIterableDataset(datapipe, n_samples_per_batch=1024),
        samples_per_gpu=batch_size,
    )

    for sample in inference_loader[0]:
        for k in keys:
            assert isinstance(sample[k], np.ndarray)
            assert batch_size == sample[k].shape[0]
            assert sample[k].shape[1] == 1024
        break
