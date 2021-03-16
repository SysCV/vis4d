"""Builders for openMT components."""
from openmt.config import Config, Matcher, RoIHead, Sampler
from openmt.core.bbox.matchers import BaseMatcher
from openmt.core.bbox.samplers import BaseSampler

# from openmt.modeling.roi_heads import BaseRoIHead
# from openmt.modeling.meta_arch import BaseMetaArch  TODO typing causes circular import
from .registry import RegistryHolder


def build_model(cfg: Config):
    """
    Build the whole model architecture using meta_arch templates.
    Note that it does not load any weights from ``cfg``.
    """
    model_registry = RegistryHolder.get_registry("modeling.meta_arch")
    print(model_registry)  # TODO fix classes wrongly not appearing in registry
    if cfg.tracking.type in model_registry:
        return model_registry[cfg.tracking.type](cfg)
    else:
        raise NotImplementedError(
            f"Meta architecture {cfg.tracking.type} not " f"found."
        )
    # TODO where should the model be moved to gpu? Here?


def build_sampler(cfg: Sampler) -> BaseSampler:
    """Build a bounding box sampler from config."""
    model_registry = RegistryHolder.get_registry("core.bbox.samplers")
    if cfg.type in model_registry:
        return model_registry[cfg.type](cfg)
    else:
        raise NotImplementedError(f"Sampler {cfg.type} not found.")


def build_matcher(cfg: Matcher) -> BaseMatcher:
    """Build a bounding box matcher from config."""
    model_registry = RegistryHolder.get_registry("core.bbox.matchers")
    if cfg.type in model_registry:
        return model_registry[cfg.type](cfg)
    else:
        raise NotImplementedError(f"Matcher {cfg.type} not found.")


def build_roi_head(cfg: RoIHead):
    """Build an RoIHead from config."""
    model_registry = RegistryHolder.get_registry("modeling.roi_heads")
    if cfg.type in model_registry:
        return model_registry[cfg.type](cfg)
    else:
        raise NotImplementedError(f"RoIHead {cfg.type} not found.")
