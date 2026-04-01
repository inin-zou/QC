"""Config file parsing and pipeline construction for RobotQ.

Provides:
- REGISTRY: mapping of transform name strings to classes
- build_pipeline: construct a Compose pipeline from a parsed YAML config dict
- load_config: read a YAML file and return the parsed dict
- resolve_adapter: map adapter name strings to adapter instances
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from robotq.core.augmentations.color import ColorJitter
from robotq.core.augmentations.mirror import Mirror
from robotq.core.augmentations.noise import ActionNoise, GaussianNoise
from robotq.core.augmentations.speed import SpeedWarp
from robotq.core.pipeline import Compose, OneOf, SomeOf
from robotq.core.transform import RobotTransform

REGISTRY: dict[str, type] = {
    "Mirror": Mirror,
    "ColorJitter": ColorJitter,
    "GaussianNoise": GaussianNoise,
    "ActionNoise": ActionNoise,
    "SpeedWarp": SpeedWarp,
}

# Lazy-register BackgroundReplace to avoid importing torch at module load time
try:
    from robotq.core.augmentations.background import BackgroundReplace

    REGISTRY["BackgroundReplace"] = BackgroundReplace
except ImportError:
    pass  # generative deps not installed

_COMPOSITE_TYPES = {"OneOf", "SomeOf"}


def _build_transform(item: dict[str, Any], adapter: Any) -> Any:
    """Build a single transform (or composite) from a config dict entry."""
    item = dict(item)  # shallow copy so we can pop without mutating caller's data
    type_name: str = item.pop("type")

    if type_name in _COMPOSITE_TYPES:
        child_configs = item.pop("transforms", [])
        children = [_build_transform(c, adapter) for c in child_configs]
        if type_name == "OneOf":
            return OneOf(children, **item)
        else:  # SomeOf
            # n may be specified as a list in YAML; convert to tuple
            if "n" in item and isinstance(item["n"], list):
                item["n"] = tuple(item["n"])
            return SomeOf(children, **item)

    if type_name not in REGISTRY:
        raise ValueError(
            f"Unknown transform type: {type_name!r}. Available types: {sorted(REGISTRY.keys())}"
        )

    cls = REGISTRY[type_name]

    if issubclass(cls, RobotTransform):
        if adapter is None:
            raise ValueError(
                f"Transform {type_name!r} is a RobotTransform and requires an adapter, "
                "but none was provided."
            )
        return cls(adapter=adapter, **item)

    return cls(**item)


def build_pipeline(config: dict, adapter: Any = None) -> Compose:
    """Build a Compose pipeline from a parsed YAML config dict.

    Parameters
    ----------
    config:
        Parsed YAML dict.  Must contain a ``"pipeline"`` key whose value is a
        list of transform specification dicts.
    adapter:
        An :class:`~robotq.adapters.base.ActionAdapter` instance, or ``None``.
        Required when any transform in the pipeline is a
        :class:`~robotq.core.transform.RobotTransform`.

    Returns
    -------
    Compose
        A :class:`~robotq.core.pipeline.Compose` wrapping all requested transforms.
    """
    transforms = [_build_transform(item, adapter) for item in config["pipeline"]]
    return Compose(transforms)


def load_config(path: str | Path) -> dict:
    """Read a YAML file and return the parsed dict.

    Parameters
    ----------
    path:
        Filesystem path to the YAML configuration file.

    Returns
    -------
    dict
        The top-level mapping parsed from the YAML file.
    """
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def resolve_adapter(name: str) -> Any:
    """Map an adapter name string to an adapter instance.

    Parameters
    ----------
    name:
        Short name for the adapter (e.g. ``"aloha"``).

    Returns
    -------
    ActionAdapter
        A freshly constructed adapter instance.

    Raises
    ------
    ValueError
        If *name* is not a recognised adapter name.
    """
    if name == "aloha":
        from robotq.adapters.aloha import AlohaAdapter

        return AlohaAdapter()

    raise ValueError(f"Unknown adapter name: {name!r}. Supported adapters: ['aloha']")
