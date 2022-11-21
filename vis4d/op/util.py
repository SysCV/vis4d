"""Utilities for op."""
from __future__ import annotations

import os
import re
import tempfile
from io import BytesIO
from typing import Any
from urllib.request import urlopen
from zipfile import ZipFile

import requests
import torch
from torch import Tensor, nn

from vis4d.common import DictStrAny
from vis4d.common.imports import MMCV_AVAILABLE

if MMCV_AVAILABLE:
    from mmcv import Config as MMConfig
    from mmcv.runner import load_checkpoint
else:
    MMConfig = None


BDD100K_MODEL_PREFIX = "https://dl.cv.ethz.ch/bdd100k/"
MM_MODEL_MAP = {
    "mmdet://": "https://download.openmmlab.com/mmdetection/v2.0/",
    "mmseg://": "https://download.openmmlab.com/mmsegmentation/v0.5/",
}
MM_CFG_MAP = {
    "mmdet://": "syscv/mmdetection/master/configs/",
    "mmseg://": "open-mmlab/mmsegmentation/master/configs/",
}
MM_ZIP_MAP = {
    "mmdet://": "mmdetection-master/configs/",
    "mmseg://": "mmsegmentation-master/configs/",
}


def unmap(data: Tensor, count: int, inds: Tensor, fill: int = 0) -> Tensor:
    """Unmap a subset of data back to the original data (of size count).

    Args:
        data (Tensor): Subset of the original data.
        count (int): Length of the original data.
        inds (Tensor): Indices of the subset entries in the original set.
        fill (int, optional): Fill value for other entries. Defaults to 0.

    Returns:
        Tensor: Tensor sized like original data that contains the subset.
    """
    if data.dim() == 1:
        ret = data.new_full((count,), fill)
        ret[inds.type(torch.bool)] = data
    else:
        new_size = (count,) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds.type(torch.bool), :] = data
    return ret


# TODO check if still needed, revise or remove
def load_model_checkpoint(
    model: nn.Module,
    weights: str,
    strict: bool = False,
    rev_keys: None | list[tuple[str, str]] = None,
) -> None:
    """Load MM model checkpoint."""
    if rev_keys is None:  # pragma: no cover
        rev_keys = [(r"^module\.", "")]
    if re.compile(r"^mm(det|seg)://").search(weights):
        pre = weights[:8]
        weights = MM_MODEL_MAP[pre] + weights.split(pre)[-1]
        load_checkpoint(model, weights, strict=strict, revise_keys=rev_keys)
    elif weights.startswith("bdd100k://"):
        weights = BDD100K_MODEL_PREFIX + weights.split("bdd100k://")[-1]
        load_checkpoint(model, weights, strict=strict, revise_keys=rev_keys)
    else:  # pragma: no cover
        load_checkpoint(model, weights, strict=strict, revise_keys=rev_keys)


def load_config_from_mm(url: str, mm_base: str) -> str:
    """Get config from mmdetection GitHub repository."""
    full_url = "https://raw.githubusercontent.com/" + mm_base + url
    response = requests.get(full_url)
    assert (
        response.status_code == 200
    ), f"Request to {full_url} failed with code {response.status_code}!"
    return response.text


def load_config(path: str, key: str = "model") -> MMConfig:
    """Load config either from file or from URL."""
    if os.path.exists(path):
        cfg = MMConfig.fromfile(path)
    elif re.compile(r"^mm(det|seg)://").search(path):
        pre = path[:8]
        cfg_content = load_config_from_mm(path.split(pre)[-1], MM_CFG_MAP[pre])
        if cfg_content.find("_base_") >= 0:
            cwd = os.getcwd()
            with tempfile.TemporaryDirectory() as temp_config_dir:
                # download configs
                url_tmp = MM_CFG_MAP[pre].replace("master/configs/", "archive")
                with ZipFile(
                    BytesIO(
                        urlopen(
                            f"https://github.com/{url_tmp}/master.zip"
                        ).read()
                    )
                ) as zipfile:
                    zipfile.extractall(path=temp_config_dir)
                os.chdir(os.path.join(temp_config_dir, MM_ZIP_MAP[pre]))
                cfg = MMConfig.fromfile(path.replace(pre, ""))
                os.chdir(cwd)
        else:
            cfg = MMConfig.fromstring(cfg_content, os.path.splitext(path)[1])
    else:
        raise FileNotFoundError(f"MM config not found: {path}")
    assert key in cfg
    return cfg[key]


def set_attr(  # type: ignore
    attr: Any, partial_keys: list[str], last_key: str, value: Any
) -> None:
    """Set specific attribute in config."""
    for i, part_k in enumerate(partial_keys):
        if isinstance(attr, list):  # pragma: no cover
            for attr_item in attr:
                set_attr(attr_item, partial_keys[i:], last_key, value)
            return
        attr = attr.get(part_k)
    attr[last_key] = value


def add_keyword_args(model_kwargs: DictStrAny, cfg: MMConfig) -> None:
    """Add keyword args in config."""
    for k, v in model_kwargs.items():
        partial_keys = k.split(".")
        partial_keys, last_key = partial_keys[:-1], partial_keys[-1]
        set_attr(cfg, partial_keys, last_key, v)
