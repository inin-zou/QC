"""Unit tests for robotq.core.episode."""

from __future__ import annotations

import numpy as np
import pytest

from robotq.core.episode import Episode, EpisodeMetadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metadata(**overrides) -> EpisodeMetadata:
    defaults = dict(
        episode_index=0,
        task_description="Pick up the cube",
        task_id=1,
        fps=30.0,
        camera_names=["top", "wrist"],
        robot_type="so100",
    )
    defaults.update(overrides)
    return EpisodeMetadata(**defaults)


def _make_frames(cameras: list[str], T: int, H: int = 4, W: int = 4) -> dict:
    return {cam: [np.zeros((H, W, 3), dtype=np.uint8) for _ in range(T)] for cam in cameras}


def _make_episode(
    T: int = 10, action_dim: int = 6, state_dim: int = 12, cameras=("top", "wrist")
) -> Episode:
    return Episode(
        frames=_make_frames(list(cameras), T),
        actions=np.zeros((T, action_dim), dtype=np.float32),
        states=np.zeros((T, state_dim), dtype=np.float32),
        metadata=_make_metadata(camera_names=list(cameras)),
    )


# ---------------------------------------------------------------------------
# Test: create valid Episode and verify all fields are accessible
# ---------------------------------------------------------------------------


def test_create_valid_episode():
    ep = _make_episode()

    assert isinstance(ep.frames, dict)
    assert set(ep.frames.keys()) == {"top", "wrist"}
    assert len(ep.frames["top"]) == 10
    assert isinstance(ep.actions, np.ndarray)
    assert ep.actions.shape == (10, 6)
    assert isinstance(ep.states, np.ndarray)
    assert ep.states.shape == (10, 12)
    assert isinstance(ep.metadata, EpisodeMetadata)
    assert ep.metadata.task_description == "Pick up the cube"
    assert ep.metadata.fps == 30.0
    assert ep.metadata.robot_type == "so100"
    assert ep.metadata.episode_index == 0
    assert ep.metadata.task_id == 1
    assert ep.metadata.camera_names == ["top", "wrist"]


# ---------------------------------------------------------------------------
# Test: num_frames, action_dim, state_dim properties
# ---------------------------------------------------------------------------


def test_properties():
    ep = _make_episode(T=15, action_dim=7, state_dim=9)

    assert ep.num_frames == 15
    assert ep.action_dim == 7
    assert ep.state_dim == 9


def test_properties_single_step():
    ep = _make_episode(T=1, action_dim=4, state_dim=3)

    assert ep.num_frames == 1
    assert ep.action_dim == 4
    assert ep.state_dim == 3


# ---------------------------------------------------------------------------
# Test: mismatched frame counts raises ValueError
# ---------------------------------------------------------------------------


def test_mismatched_frame_counts_raises():
    T = 10
    frames = {
        "top": [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(T)],
        "wrist": [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(T + 3)],
    }
    with pytest.raises(ValueError, match="Frame counts differ across cameras"):
        Episode(
            frames=frames,
            actions=np.zeros((T, 6), dtype=np.float32),
            states=np.zeros((T, 12), dtype=np.float32),
            metadata=_make_metadata(),
        )


def test_frames_actions_mismatch_raises():
    T = 10
    with pytest.raises(ValueError, match="actions T dimension"):
        Episode(
            frames=_make_frames(["top"], T),
            actions=np.zeros((T + 5, 6), dtype=np.float32),
            states=np.zeros((T, 12), dtype=np.float32),
            metadata=_make_metadata(camera_names=["top"]),
        )


def test_frames_states_mismatch_raises():
    T = 10
    with pytest.raises(ValueError, match="states T dimension"):
        Episode(
            frames=_make_frames(["top"], T),
            actions=np.zeros((T, 6), dtype=np.float32),
            states=np.zeros((T + 2, 12), dtype=np.float32),
            metadata=_make_metadata(camera_names=["top"]),
        )


# ---------------------------------------------------------------------------
# Test: metadata extra field works
# ---------------------------------------------------------------------------


def test_metadata_extra_field_default_empty():
    meta = _make_metadata()
    assert meta.extra == {}


def test_metadata_extra_field_accepts_arbitrary_data():
    extra = {"augmented": True, "source_index": 42, "tags": ["mirror", "speed"]}
    meta = _make_metadata(extra=extra)

    assert meta.extra["augmented"] is True
    assert meta.extra["source_index"] == 42
    assert meta.extra["tags"] == ["mirror", "speed"]


def test_metadata_extra_field_isolation():
    """Two EpisodeMetadata instances must not share the same extra dict."""
    meta1 = _make_metadata()
    meta2 = _make_metadata()
    meta1.extra["key"] = "value"

    assert "key" not in meta2.extra


def test_episode_with_extra_metadata():
    ep = _make_episode()
    ep.metadata.extra["noise_scale"] = 0.05
    assert ep.metadata.extra["noise_scale"] == pytest.approx(0.05)
