"""Tests for BackgroundReplace augmentation."""

import numpy as np
import pytest

from robotq.core.augmentations.background import BackgroundReplace
from robotq.core.episode import Episode, EpisodeMetadata
from robotq.core.transform import TrajectoryTransform


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_episode(
    n_frames: int = 10,
    height: int = 64,
    width: int = 64,
    action_dim: int = 4,
    state_dim: int = 4,
) -> Episode:
    frames = [
        np.random.randint(50, 200, (height, width, 3), dtype=np.uint8) for _ in range(n_frames)
    ]
    return Episode(
        frames={"cam": frames},
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


def _make_episode_with_moving_object(
    n_frames: int = 15,
    height: int = 128,
    width: int = 128,
) -> Episode:
    """Create an episode with a bright square that moves across a static background.

    The background is a uniform gray field. A 24x24 white square shifts 4 pixels
    to the right per frame, so only frames that show the square in different
    positions will differ — giving the median subtraction a clear foreground signal.

    The frame size must be large enough that the foreground blob survives the
    15x15 morphological open applied inside _compute_mask_fast.
    """
    bg_value = 80
    sq_value = 240
    sq_size = 24

    frames = []
    for i in range(n_frames):
        frame = np.full((height, width, 3), bg_value, dtype=np.uint8)
        x = (i * 4) % (width - sq_size)
        y = height // 4
        frame[y : y + sq_size, x : x + sq_size] = sq_value
        frames.append(frame)

    return Episode(
        frames={"cam": frames},
        actions=np.zeros((n_frames, 4), dtype=np.float32),
        states=np.zeros((n_frames, 4), dtype=np.float32),
        metadata=EpisodeMetadata(
            episode_index=0,
            task_description="test",
            task_id=0,
            fps=30.0,
            camera_names=["cam"],
            robot_type="test",
        ),
    )


def _make_static_episode(
    n_frames: int = 10,
    height: int = 64,
    width: int = 64,
) -> Episode:
    """Create an episode where every frame is identical (no motion)."""
    single_frame = np.full((height, width, 3), 100, dtype=np.uint8)
    frames = [single_frame.copy() for _ in range(n_frames)]
    return Episode(
        frames={"cam": frames},
        actions=np.zeros((n_frames, 4), dtype=np.float32),
        states=np.zeros((n_frames, 4), dtype=np.float32),
        metadata=EpisodeMetadata(
            episode_index=0,
            task_description="test",
            task_id=0,
            fps=30.0,
            camera_names=["cam"],
            robot_type="test",
        ),
    )


# ---------------------------------------------------------------------------
# _compute_mask_fast tests
# ---------------------------------------------------------------------------


def test_compute_mask_fast_returns_correct_shape():
    """Mask shape must match the HxW of the input frames."""
    height, width = 48, 80
    ep = _make_episode(n_frames=10, height=height, width=width)
    bg = BackgroundReplace(prompt="test", method="fast", p=1.0)
    mask = bg._compute_mask_fast(ep.frames["cam"])
    assert mask.shape == (height, width), (
        f"Expected mask shape ({height}, {width}), got {mask.shape}"
    )


def test_compute_mask_fast_detects_foreground():
    """A moving bright square against a static background should produce both
    foreground pixels (0) and background pixels (255) in the mask."""
    ep = _make_episode_with_moving_object(n_frames=15)
    bg = BackgroundReplace(prompt="test", method="fast", p=1.0)
    mask = bg._compute_mask_fast(ep.frames["cam"])

    n_foreground = int((mask == 0).sum())
    n_background = int((mask == 255).sum())

    assert n_foreground > 0, "Expected some foreground (0) pixels but found none."
    assert n_background > 0, "Expected some background (255) pixels but found none."


def test_compute_mask_fast_static_scene_is_all_background():
    """Identical frames (no motion) should produce a mask that is entirely or
    almost entirely background (255), since nothing differs from the median."""
    ep = _make_static_episode(n_frames=10)
    bg = BackgroundReplace(prompt="test", method="fast", p=1.0)
    mask = bg._compute_mask_fast(ep.frames["cam"])

    total_pixels = mask.size
    background_pixels = int((mask == 255).sum())
    background_ratio = background_pixels / total_pixels

    # Allow up to 5 % foreground from morphological artefacts at image edges
    assert background_ratio >= 0.95, (
        f"Expected >= 95 % background pixels, got {background_ratio:.1%}"
    )


# ---------------------------------------------------------------------------
# Construction / validation tests
# ---------------------------------------------------------------------------


def test_background_replace_is_trajectory_transform():
    """BackgroundReplace must be an instance of TrajectoryTransform."""
    bg = BackgroundReplace(prompt="lab", method="fast", p=1.0)
    assert isinstance(bg, TrajectoryTransform)


def test_background_replace_invalid_method():
    """Passing an unsupported method string must raise ValueError."""
    with pytest.raises(ValueError, match="method must be"):
        BackgroundReplace(prompt="lab", method="invalid")


def test_background_replace_p_validation():
    """Probability p outside [0, 1] must raise ValueError (inherited from Transform)."""
    with pytest.raises(ValueError):
        BackgroundReplace(prompt="lab", method="fast", p=2.0)


# ---------------------------------------------------------------------------
# apply_to_episode with mocked inpaint pipe
# ---------------------------------------------------------------------------


def test_background_replace_preserves_actions():
    """With a no-op inpaint pipe the output actions must be identical to the input."""
    ep = _make_episode_with_moving_object(n_frames=15)
    original_actions = ep.actions.copy()

    bg = BackgroundReplace(prompt="test", method="fast", p=1.0)

    # Mock the inpaint pipe to return the input image unchanged
    class MockPipe:
        def __call__(self, **kwargs):
            class Result:
                images = [kwargs["image"]]

            return Result()

    bg._pipe = MockPipe()

    result = bg.apply_to_episode(ep)

    np.testing.assert_array_equal(
        result.actions,
        original_actions,
        err_msg="Actions must not be modified by BackgroundReplace.",
    )
