"""Builders for openMT components."""
import torch

import openmt.core.bbox.matchers as matchers
import openmt.core.bbox.samplers as samplers
import openmt.modeling.meta_arch as meta_arch
import openmt.modeling.roi_heads as roi_heads
from openmt.config import Config, ...  # TODO implement all the configs


def build_model(cfg: Config) -> torch.nn.Module:
    """
    Build the whole model architecture using meta_arch templates.
    Note that it does not load any weights from ``cfg``.
    """
    if hasattr(meta_arch, cfg.type):
        args = cfg.__dict__
        del args["name"]
        return getattr(meta_arch, cfg.type)(args)
    else:
        raise NotImplementedError(f"Meta architecture {cfg.type} not found.")
    # TODO where should the model be moved to gpu? Here?


def build_sampler(cfg: SamplerConfig) -> samplers.BaseSampler:
    """Build a bounding box sampler from config."""
    if hasattr(samplers, cfg.type):
        args = cfg.__dict__
        del args["name"]
        return getattr(samplers, cfg.type)(args)
    else:
        raise NotImplementedError(f"Sampler {cfg.type} not found.")


def build_matcher(cfg: MatcherConfig) -> matchers.BaseMatcher:
    """Build a bounding box matcher from config."""
    if hasattr(matchers, cfg.type):
        args = cfg.__dict__
        del args["name"]
        return getattr(matchers, cfg.type)(args)
    else:
        raise NotImplementedError(f"Matcher {cfg.type} not found.")


def build_roihead(cfg: RoIHeadConfig) -> roi_heads.BaseRoIHead:
    """Build an RoIHead from config."""
    if hasattr(roi_heads, cfg.type):
        args = cfg.__dict__
        del args["name"]
        return getattr(roi_heads, cfg.type)(args)
    else:
        raise NotImplementedError(f"RoIHead {cfg.type} not found.")