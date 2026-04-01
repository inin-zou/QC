"""Tests for GaussianNoise and ActionNoise augmentations."""

import numpy as np
import pytest

from robotq.core.augmentations.noise import ActionNoise, GaussianNoise
from robotq.core.episode import Episode, EpisodeMetadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_episode(n_frames: int = 5, action_dim: int = 4, state_dim: int = 4) -> Episode:
    return Episode(
        frames={
            "cam": [
                np.random.randint(50, 200, (32, 32, 3), dtype=np.uint8)
                for _ in range(n_frames)
            ]
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
# GaussianNoise tests
# ---------------------------------------------------------------------------

def test_gaussian_noise_changes_pixel_values():
    """Applying GaussianNoise with non-zero sigma should change at least some pixels."""
    ep = _make_episode()
    aug = GaussianNoise(sigma=0.1, p=1.0)
    result = aug(ep)
    any_different = any(
        not np.array_equal(orig, aug_frame)
        for orig, aug_frame in zip(ep.frames["cam"], result.frames["cam"])
    )
    assert any_different


def test_gaussian_noise_sigma_zero_leaves_frames_unchanged():
    """With sigma=0 the noise term is zero so frames must be identical."""
    ep = _make_episode()
    aug = GaussianNoise(sigma=0.0, p=1.0)
    result = aug(ep)
    for orig, out in zip(ep.frames["cam"], result.frames["cam"]):
        np.testing.assert_array_equal(orig, out)


def test_gaussian_noise_preserves_shape_and_dtype():
    """Output frames must have the same shape and uint8 dtype as input frames."""
    ep = _make_episode()
    aug = GaussianNoise(sigma=0.05, p=1.0)
    result = aug(ep)
    for orig, out in zip(ep.frames["cam"], result.frames["cam"]):
        assert out.shape == orig.shape
        assert out.dtype == np.uint8


def test_gaussian_noise_clips_to_valid_range():
    """All output pixel values must be in [0, 255]."""
    # Use extreme pixel values to exercise clipping paths
    frames = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(5)]
    frames += [np.full((32, 32, 3), 255, dtype=np.uint8) for _ in range(5)]
    ep = Episode(
        frames={"cam": frames},
        actions=np.zeros((10, 4), dtype=np.float32),
        states=np.zeros((10, 4), dtype=np.float32),
        metadata=EpisodeMetadata(
            episode_index=0, task_description="test", task_id=0,
            fps=50.0, camera_names=["cam"], robot_type="test",
        ),
    )
    aug = GaussianNoise(sigma=1.0, p=1.0)  # very large sigma to force clipping
    result = aug(ep)
    for out in result.frames["cam"]:
        assert out.min() >= 0
        assert out.max() <= 255


def test_gaussian_noise_each_frame_gets_independent_noise():
    """Uniform frames should produce different outputs per frame (independent noise)."""
    frames = [np.full((32, 32, 3), 128, dtype=np.uint8) for _ in range(10)]
    ep = Episode(
        frames={"cam": frames},
        actions=np.zeros((10, 2), dtype=np.float32),
        states=np.zeros((10, 2), dtype=np.float32),
        metadata=EpisodeMetadata(
            episode_index=0, task_description="test", task_id=0,
            fps=50.0, camera_names=["cam"], robot_type="test",
        ),
    )
    aug = GaussianNoise(sigma=0.1, p=1.0)
    result = aug(ep)
    # With independent noise the frames should differ from each other
    first = result.frames["cam"][0]
    any_different = any(
        not np.array_equal(first, f) for f in result.frames["cam"][1:]
    )
    assert any_different


# ---------------------------------------------------------------------------
# ActionNoise tests
# ---------------------------------------------------------------------------

def test_action_noise_changes_action_values():
    """ActionNoise with non-zero sigma must change at least some action values."""
    ep = _make_episode()
    aug = ActionNoise(sigma=0.1, p=1.0)
    result = aug(ep)
    assert not np.array_equal(ep.actions, result.actions)


def test_action_noise_leaves_frames_untouched():
    """ActionNoise must not modify frames."""
    ep = _make_episode()
    orig_frames = [f.copy() for f in ep.frames["cam"]]
    aug = ActionNoise(sigma=0.1, p=1.0)
    result = aug(ep)
    for orig, out in zip(orig_frames, result.frames["cam"]):
        np.testing.assert_array_equal(orig, out)


def test_action_noise_leaves_states_untouched():
    """ActionNoise must not modify states."""
    ep = _make_episode()
    aug = ActionNoise(sigma=0.1, p=1.0)
    result = aug(ep)
    np.testing.assert_array_equal(ep.states, result.states)


def test_action_noise_output_actions_same_shape():
    """Output actions must have the same shape as input actions."""
    ep = _make_episode(n_frames=8, action_dim=7)
    aug = ActionNoise(sigma=0.05, p=1.0)
    result = aug(ep)
    assert result.actions.shape == ep.actions.shape


def test_action_noise_does_not_mutate_input():
    """The original episode's actions must remain unchanged after augmentation."""
    ep = _make_episode()
    original_actions = ep.actions.copy()
    aug = ActionNoise(sigma=0.5, p=1.0)
    aug(ep)
    np.testing.assert_array_equal(ep.actions, original_actions)


def test_action_noise_returns_new_episode():
    """apply_to_episode must return a new Episode object, not the same one."""
    ep = _make_episode()
    aug = ActionNoise(sigma=0.1, p=1.0)
    result = aug(ep)
    assert result is not ep
