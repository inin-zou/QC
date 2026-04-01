"""Tests for SpeedWarp augmentation."""

from __future__ import annotations

import numpy as np
import pytest

from robotq.core.augmentations.speed import SpeedWarp
from robotq.core.episode import Episode, EpisodeMetadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_episode(n_frames: int = 10, action_dim: int = 4, state_dim: int = 6) -> Episode:
    return Episode(
        frames={
            "cam": [np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8) for _ in range(n_frames)]
        },
        actions=np.random.randn(n_frames, action_dim).astype(np.float32),
        states=np.random.randn(n_frames, state_dim).astype(np.float32),
        metadata=EpisodeMetadata(
            episode_index=0,
            task_description="test",
            task_id=0,
            fps=50.0,
            camera_names=["cam"],
            robot_type="test",
        ),
    )


# ---------------------------------------------------------------------------
# Core behaviour tests
# ---------------------------------------------------------------------------


def test_speed_up_halves_frame_count():
    """rate=2.0 should approximately halve the number of frames."""
    ep = _make_episode(n_frames=20)
    aug = SpeedWarp(min_rate=2.0, max_rate=2.0, p=1.0)
    result = aug(ep)
    assert result.num_frames == round(20 / 2.0)


def test_slow_down_doubles_frame_count():
    """rate=0.5 should approximately double the number of frames."""
    ep = _make_episode(n_frames=10)
    aug = SpeedWarp(min_rate=0.5, max_rate=0.5, p=1.0)
    result = aug(ep)
    assert result.num_frames == round(10 / 0.5)


def test_identity_rate_preserves_length():
    """rate=1.0 must return an episode with the same number of frames."""
    ep = _make_episode(n_frames=12)
    aug = SpeedWarp(min_rate=1.0, max_rate=1.0, p=1.0)
    result = aug(ep)
    assert result.num_frames == ep.num_frames


def test_preserves_action_dim():
    """The action dimension must be unchanged after resampling."""
    ep = _make_episode(n_frames=8, action_dim=7)
    aug = SpeedWarp(min_rate=0.5, max_rate=2.0, p=1.0)
    result = aug(ep)
    assert result.action_dim == ep.action_dim


def test_preserves_state_dim():
    """The state dimension must be unchanged after resampling."""
    ep = _make_episode(n_frames=8, state_dim=5)
    aug = SpeedWarp(min_rate=0.5, max_rate=2.0, p=1.0)
    result = aug(ep)
    assert result.state_dim == ep.state_dim


def test_returns_new_episode():
    """apply_to_episode must return a new Episode object, not the same one."""
    ep = _make_episode()
    aug = SpeedWarp(min_rate=1.0, max_rate=1.0, p=1.0)
    result = aug(ep)
    assert result is not ep


def test_does_not_mutate_original_actions():
    """The original episode actions must remain unchanged after augmentation."""
    ep = _make_episode(n_frames=10)
    original_actions = ep.actions.copy()
    aug = SpeedWarp(min_rate=2.0, max_rate=2.0, p=1.0)
    aug(ep)
    np.testing.assert_array_equal(ep.actions, original_actions)


def test_does_not_mutate_original_frames():
    """The original episode frames must remain unchanged after augmentation."""
    ep = _make_episode(n_frames=10)
    original_frames = [f.copy() for f in ep.frames["cam"]]
    aug = SpeedWarp(min_rate=2.0, max_rate=2.0, p=1.0)
    aug(ep)
    for orig, out in zip(original_frames, ep.frames["cam"]):
        np.testing.assert_array_equal(orig, out)


# ---------------------------------------------------------------------------
# Consistency checks
# ---------------------------------------------------------------------------


def test_actions_shape_consistent_with_frames():
    """actions.shape[0] must equal the number of frames in each camera."""
    ep = _make_episode(n_frames=15)
    aug = SpeedWarp(min_rate=1.5, max_rate=1.5, p=1.0)
    result = aug(ep)
    assert result.actions.shape[0] == len(result.frames["cam"])


def test_states_shape_consistent_with_frames():
    """states.shape[0] must equal the number of frames in each camera."""
    ep = _make_episode(n_frames=15)
    aug = SpeedWarp(min_rate=0.75, max_rate=0.75, p=1.0)
    result = aug(ep)
    assert result.states.shape[0] == len(result.frames["cam"])


def test_output_actions_dtype_preserved():
    """Output actions must keep float32 dtype."""
    ep = _make_episode(n_frames=10)
    aug = SpeedWarp(min_rate=2.0, max_rate=2.0, p=1.0)
    result = aug(ep)
    assert result.actions.dtype == np.float32


def test_output_states_dtype_preserved():
    """Output states must keep float32 dtype."""
    ep = _make_episode(n_frames=10)
    aug = SpeedWarp(min_rate=2.0, max_rate=2.0, p=1.0)
    result = aug(ep)
    assert result.states.dtype == np.float32


# ---------------------------------------------------------------------------
# Invalid parameter tests
# ---------------------------------------------------------------------------


def test_invalid_min_rate_zero_raises():
    with pytest.raises(ValueError):
        SpeedWarp(min_rate=0.0, max_rate=1.0)


def test_invalid_min_rate_negative_raises():
    with pytest.raises(ValueError):
        SpeedWarp(min_rate=-0.5, max_rate=1.0)


def test_invalid_max_rate_less_than_min_raises():
    with pytest.raises(ValueError):
        SpeedWarp(min_rate=1.5, max_rate=1.0)


def test_p_out_of_range_raises():
    with pytest.raises(ValueError):
        SpeedWarp(min_rate=0.8, max_rate=1.2, p=1.5)
