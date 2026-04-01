"""Tests for robotq/core/config.py."""

from __future__ import annotations

import pytest

from robotq.adapters.aloha import AlohaAdapter
from robotq.core.augmentations.color import ColorJitter
from robotq.core.augmentations.mirror import Mirror
from robotq.core.augmentations.noise import GaussianNoise
from robotq.core.config import build_pipeline, load_config, resolve_adapter
from robotq.core.pipeline import Compose, OneOf


# ---------------------------------------------------------------------------
# build_pipeline
# ---------------------------------------------------------------------------


def test_build_pipeline_single_color_jitter():
    """build_pipeline with one ColorJitter returns a Compose with 1 transform."""
    config = {
        "pipeline": [
            {"type": "ColorJitter", "brightness": 0.2},
        ]
    }
    pipeline = build_pipeline(config)
    assert isinstance(pipeline, Compose)
    assert len(pipeline.transforms) == 1
    assert isinstance(pipeline.transforms[0], ColorJitter)


def test_build_pipeline_mirror_with_adapter():
    """build_pipeline with Mirror and an adapter injects the adapter."""
    adapter = AlohaAdapter()
    config = {
        "pipeline": [
            {"type": "Mirror", "p": 0.5},
        ]
    }
    pipeline = build_pipeline(config, adapter=adapter)
    assert isinstance(pipeline, Compose)
    assert len(pipeline.transforms) == 1
    mirror = pipeline.transforms[0]
    assert isinstance(mirror, Mirror)
    assert mirror.adapter is adapter
    assert mirror.p == 0.5


def test_build_pipeline_mirror_without_adapter_raises():
    """build_pipeline with Mirror but no adapter raises ValueError."""
    config = {
        "pipeline": [
            {"type": "Mirror"},
        ]
    }
    with pytest.raises(ValueError, match="adapter"):
        build_pipeline(config)


def test_build_pipeline_oneof_builds_correct_children():
    """build_pipeline with OneOf constructs OneOf with the right children."""
    config = {
        "pipeline": [
            {
                "type": "OneOf",
                "p": 0.8,
                "transforms": [
                    {"type": "ColorJitter", "brightness": 0.3},
                    {"type": "GaussianNoise", "sigma": 0.02},
                ],
            }
        ]
    }
    pipeline = build_pipeline(config)
    assert isinstance(pipeline, Compose)
    assert len(pipeline.transforms) == 1
    one_of = pipeline.transforms[0]
    assert isinstance(one_of, OneOf)
    assert one_of.p == 0.8
    assert len(one_of.transforms) == 2
    assert isinstance(one_of.transforms[0], ColorJitter)
    assert isinstance(one_of.transforms[1], GaussianNoise)


def test_build_pipeline_unknown_type_raises():
    """build_pipeline with an unrecognised type name raises ValueError."""
    config = {
        "pipeline": [
            {"type": "NonExistentTransform"},
        ]
    }
    with pytest.raises(ValueError, match="NonExistentTransform"):
        build_pipeline(config)


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


def test_load_config_reads_yaml(tmp_path):
    """load_config parses a YAML file and returns the expected dict."""
    yaml_content = """\
dataset: lerobot/test
adapter: aloha
pipeline:
  - type: ColorJitter
    brightness: 0.3
"""
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(yaml_content)

    cfg = load_config(yaml_file)
    assert cfg["dataset"] == "lerobot/test"
    assert cfg["adapter"] == "aloha"
    assert len(cfg["pipeline"]) == 1
    assert cfg["pipeline"][0]["type"] == "ColorJitter"
    assert cfg["pipeline"][0]["brightness"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# resolve_adapter
# ---------------------------------------------------------------------------


def test_resolve_adapter_aloha_returns_aloha_adapter():
    """resolve_adapter('aloha') returns an AlohaAdapter instance."""
    adapter = resolve_adapter("aloha")
    assert isinstance(adapter, AlohaAdapter)


def test_resolve_adapter_unknown_raises():
    """resolve_adapter raises ValueError for an unknown adapter name."""
    with pytest.raises(ValueError, match="unknown"):
        resolve_adapter("unknown")
