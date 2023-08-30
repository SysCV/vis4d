"""Model Zoo."""
from __future__ import annotations

from typing import Callable

from vis4d.common.typing import ArgsType

from .bdd100k import AVAILABLE_MODELS as BDD100K_MODELS
from .bevformer import AVAILABLE_MODELS as BEVFORMER_MODELS
from .cc_3dt import AVAILABLE_MODELS as CC_3DT_MODELS
from .faster_rcnn import AVAILABLE_MODELS as FASTER_RCNN_MODELS
from .fcn_resnet import AVAILABLE_MODELS as FCN_RESNET_MODELS
from .mask_rcnn import AVAILABLE_MODELS as MASK_RCNN_MODELS
from .qdtrack import AVAILABLE_MODELS as QDTRACK_MODELS
from .retinanet import AVAILABLE_MODELS as RETINANET_MODELS
from .shift import AVAILABLE_MODELS as SHIFT_MODELS
from .vit import AVAILABLE_MODELS as VIT_MODELS
from .yolox import AVAILABLE_MODELS as YOLOX_MODELS

TFunc = Callable[..., ArgsType]

AVAILABLE_MODELS = {
    "bdd100k": BDD100K_MODELS,
    "cc_3dt": CC_3DT_MODELS,
    "bevformer": BEVFORMER_MODELS,
    "faster_rcnn": FASTER_RCNN_MODELS,
    "fcn_resnet": FCN_RESNET_MODELS,
    "mask_rcnn": MASK_RCNN_MODELS,
    "qdtrack": QDTRACK_MODELS,
    "retinanet": RETINANET_MODELS,
    "shift": SHIFT_MODELS,
    "vit": VIT_MODELS,
    "yolox": YOLOX_MODELS,
}


def register_config(category: str, name: str) -> Callable[TFunc, None]:
    """Register a config in the model zoo for the given name and category.

    The config will then be available via `get_config_by_name` utilities and
    located in the AVAILABLE_MODELS dictionary located at
    [category][name].

    Args:
        category: Category of the config.
        name: Name of the config.

    Returns:
        The decorator.
    """

    def decorator(fnc_or_clazz: TFunc) -> None:
        module = fnc_or_clazz

        if callable(fnc_or_clazz):
            # Directly annotated get_config function. Wrap it and register it.
            class Wrapper:
                """Wrapper class."""

                def get_config(self, *args, **kwargs) -> ArgsType:
                    """Resolves the get_config function."""
                    return fnc_or_clazz(*args, **kwargs)

            module = Wrapper()

        # Register the config
        if category not in AVAILABLE_MODELS:
            AVAILABLE_MODELS[category] = {}

        AVAILABLE_MODELS[category][name] = module

    return decorator
