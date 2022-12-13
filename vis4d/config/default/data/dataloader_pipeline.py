from __future__ import annotations

from typing import Callable

from ml_collections.config_dict import ConfigDict

from vis4d.config.util import class_config
from vis4d.data.typing import DictData


def get_dataloader_config(
    transforms: list[ConfigDict],
    dataset: ConfigDict,
    batch_size: int = 1,
    batchprocess_fn: Callable[[list[DictData]], list[DictData]]
    | ConfigDict = lambda x: x,
) -> ConfigDict:
    return class_config(
        "vis4d.data.loader.build_train_dataloader",
        dataset=class_config(
            "vis4d.data.loader.DataPipe",
            datasets=dataset,
            preprocess_fn=class_config(
                "vis4d.data.transforms.base.compose", transforms=transforms
            ),
        ),
        batchprocess_fn=batchprocess_fn,
        samples_per_gpu=batch_size,
        workers_per_gpu=1,
    )
