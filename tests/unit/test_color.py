"""Tests for ColorJitter augmentation."""

import numpy as np

from robotq.core.augmentations.color import ColorJitter
from robotq.core.episode import Episode, EpisodeMetadata


def _make_episode(n_frames=5):
    return Episode(
        frames={
            "cam": [
                np.random.randint(50, 200, (32, 32, 3), dtype=np.uint8) for _ in range(n_frames)
            ]
        },
        actions=np.random.randn(n_frames, 4).astype(np.float32),
        states=np.random.randn(n_frames, 4).astype(np.float32),
        metadata=EpisodeMetadata(
            episode_index=0,
            task_description="test",
            task_id=0,
            fps=50.0,
            camera_names=["cam"],
            robot_type="test",
        ),
    )


def test_color_jitter_changes_pixels():
    ep = _make_episode()
    jitter = ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=1.0)
    result = jitter(ep)
    # At least some frames should differ
    different = False
    for orig, aug in zip(ep.frames["cam"], result.frames["cam"]):
        if not np.array_equal(orig, aug):
            different = True
            break
    assert different


def test_color_jitter_preserves_actions():
    ep = _make_episode()
    jitter = ColorJitter(p=1.0)
    result = jitter(ep)
    np.testing.assert_array_equal(ep.actions, result.actions)


def test_color_jitter_preserves_states():
    ep = _make_episode()
    jitter = ColorJitter(p=1.0)
    result = jitter(ep)
    np.testing.assert_array_equal(ep.states, result.states)


def test_color_jitter_preserves_shape_dtype():
    ep = _make_episode()
    jitter = ColorJitter(p=1.0)
    result = jitter(ep)
    for orig, aug in zip(ep.frames["cam"], result.frames["cam"]):
        assert aug.shape == orig.shape
        assert aug.dtype == np.uint8


def test_color_jitter_temporal_consistency():
    """All frames in an episode should get the same jitter params (no flickering)."""
    # Use uniform frames so we can detect if params differ
    frames = [np.full((32, 32, 3), 128, dtype=np.uint8) for _ in range(10)]
    ep = Episode(
        frames={"cam": frames},
        actions=np.zeros((10, 2), dtype=np.float32),
        states=np.zeros((10, 2), dtype=np.float32),
        metadata=EpisodeMetadata(
            episode_index=0,
            task_description="test",
            task_id=0,
            fps=50.0,
            camera_names=["cam"],
            robot_type="test",
        ),
    )
    jitter = ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.05, p=1.0)
    result = jitter(ep)
    # All output frames should be identical (same params applied to identical input)
    first = result.frames["cam"][0]
    for frame in result.frames["cam"][1:]:
        np.testing.assert_array_equal(first, frame)


def test_color_jitter_zero_params_approximately_unchanged():
    ep = _make_episode()
    jitter = ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0, p=1.0)
    result = jitter(ep)
    for orig, aug in zip(ep.frames["cam"], result.frames["cam"]):
        # HSV round-trip introduces small pixel differences (up to ~3)
        assert np.allclose(orig, aug, atol=4)
