"""Tests for Mirror augmentation."""

import numpy as np
import pytest

from robotq.adapters.aloha import AlohaAdapter
from robotq.core.augmentations.mirror import Mirror
from robotq.core.episode import Episode, EpisodeMetadata


def _make_episode(n_frames=5):
    """Create an episode with known asymmetric pixel pattern."""
    frames = []
    for _ in range(n_frames):
        frame = np.zeros((4, 8, 3), dtype=np.uint8)
        frame[:, :4, :] = 100  # Left half bright
        frame[:, 4:, :] = 200  # Right half brighter
        frames.append(frame)

    # Actions: left arm = 1..7, right arm = 8..14
    actions = np.tile(np.arange(1, 15, dtype=np.float32), (n_frames, 1))
    states = np.tile(np.arange(1, 15, dtype=np.float32), (n_frames, 1))

    return Episode(
        frames={"cam": frames},
        actions=actions,
        states=states,
        metadata=EpisodeMetadata(
            episode_index=0, task_description="test", task_id=0,
            fps=50.0, camera_names=["cam"], robot_type="aloha",
        ),
    )


def test_mirror_flips_frames():
    ep = _make_episode()
    mirror = Mirror(adapter=AlohaAdapter(), p=1.0)
    result = mirror(ep)
    # After flip: left half should be 200, right half 100
    frame = result.frames["cam"][0]
    assert frame[0, 0, 0] == 200  # Was right, now left
    assert frame[0, 7, 0] == 100  # Was left, now right


def test_mirror_swaps_left_right_actions():
    ep = _make_episode()
    mirror = Mirror(adapter=AlohaAdapter(), p=1.0)
    result = mirror(ep)
    # Original: left=[1..7], right=[8..14]
    # After swap: left gets [8..14], right gets [1..7]
    # Then flip_signs negate indices 0 and 7
    row = result.actions[0]
    # Index 0 (was right waist=8, negated): -8
    assert row[0] == pytest.approx(-8.0)
    # Index 1 (was right shoulder=9, not negated): 9
    assert row[1] == pytest.approx(9.0)
    # Index 7 (was left waist=1, negated): -1
    assert row[7] == pytest.approx(-1.0)
    # Index 8 (was left shoulder=2, not negated): 2
    assert row[8] == pytest.approx(2.0)


def test_mirror_swaps_states():
    ep = _make_episode()
    mirror = Mirror(adapter=AlohaAdapter(), p=1.0)
    result = mirror(ep)
    row = result.states[0]
    # Same logic as actions
    assert row[0] == pytest.approx(-8.0)
    assert row[7] == pytest.approx(-1.0)


def test_mirror_preserves_shape():
    ep = _make_episode()
    mirror = Mirror(adapter=AlohaAdapter(), p=1.0)
    result = mirror(ep)
    assert result.actions.shape == ep.actions.shape
    assert result.states.shape == ep.states.shape
    assert len(result.frames["cam"]) == len(ep.frames["cam"])


def test_mirror_does_not_mutate_original():
    ep = _make_episode()
    orig_actions = ep.actions.copy()
    mirror = Mirror(adapter=AlohaAdapter(), p=1.0)
    mirror(ep)
    np.testing.assert_array_equal(ep.actions, orig_actions)


def test_mirror_p0_unchanged():
    ep = _make_episode()
    mirror = Mirror(adapter=AlohaAdapter(), p=0.0)
    result = mirror(ep)
    assert result is ep
