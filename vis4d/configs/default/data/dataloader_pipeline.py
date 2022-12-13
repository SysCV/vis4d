from __future__ import annotations

from ml_collections.config_dict import ConfigDict

from vis4d.configs.util import class_config


def get_dataloader_config(
    transforms: list[ConfigDict], dataset: ConfigDict, batch_size: int = 1
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
        samples_per_gpu=batch_size,
        workers_per_gpu=1,
    )
