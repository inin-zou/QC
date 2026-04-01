"""Tests for pipeline composition operators: Compose, OneOf, SomeOf."""

import random

import numpy as np
import pytest

from robotq.core.episode import Episode, EpisodeMetadata
from robotq.core.pipeline import Compose, OneOf, SomeOf
from robotq.core.transform import Transform


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_episode(n_frames: int = 5) -> Episode:
    return Episode(
        frames={"cam": [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(n_frames)]},
        actions=np.zeros((n_frames, 2), dtype=np.float32),
        states=np.zeros((n_frames, 2), dtype=np.float32),
        metadata=EpisodeMetadata(
            episode_index=0,
            task_description="test",
            task_id=0,
            fps=50.0,
            camera_names=["cam"],
            robot_type="test",
        ),
    )


class MockTransform(Transform):
    def __init__(self, name: str, p: float = 1.0) -> None:
        super().__init__(p=p)
        self.name = name
        self.call_count = 0

    def apply(self, episode: Episode) -> Episode:
        if random.random() > self.p:
            return episode
        self.call_count += 1
        return episode


# ---------------------------------------------------------------------------
# Compose tests
# ---------------------------------------------------------------------------

def test_compose_applies_all_transforms_in_order():
    """All transforms in a Compose are called exactly once."""
    ep = _make_episode()
    t1 = MockTransform("a")
    t2 = MockTransform("b")
    t3 = MockTransform("c")
    result = Compose([t1, t2, t3], p=1.0)(ep)
    assert t1.call_count == 1
    assert t2.call_count == 1
    assert t3.call_count == 1
    assert result is ep  # MockTransform returns the same episode object


def test_compose_p0_returns_unchanged():
    """Compose with p=0 skips everything; no child is called."""
    ep = _make_episode()
    t1 = MockTransform("a")
    t2 = MockTransform("b")
    result = Compose([t1, t2], p=0.0)(ep)
    assert t1.call_count == 0
    assert t2.call_count == 0
    assert result is ep


def test_compose_empty_returns_episode_unchanged():
    """Compose with an empty transform list returns the episode as-is."""
    ep = _make_episode()
    result = Compose([], p=1.0)(ep)
    assert result is ep


def test_compose_repr_includes_child_transforms():
    """Compose __repr__ mentions child transform class names."""
    t1 = MockTransform("a")
    t2 = MockTransform("b")
    c = Compose([t1, t2], p=0.5)
    r = repr(c)
    assert "Compose" in r
    assert "MockTransform" in r
    assert "p=0.5" in r


# ---------------------------------------------------------------------------
# OneOf tests
# ---------------------------------------------------------------------------

def test_oneof_picks_exactly_one():
    """Over many runs, exactly one transform is called each time."""
    for _ in range(50):
        ep = _make_episode()
        t1 = MockTransform("a")
        t2 = MockTransform("b")
        t3 = MockTransform("c")
        OneOf([t1, t2, t3], p=1.0)(ep)
        total_calls = t1.call_count + t2.call_count + t3.call_count
        assert total_calls == 1, (
            f"Expected exactly 1 call, got {total_calls} "
            f"(a={t1.call_count}, b={t2.call_count}, c={t3.call_count})"
        )


def test_oneof_p0_returns_unchanged():
    """OneOf with p=0 calls no transform."""
    ep = _make_episode()
    t1 = MockTransform("a")
    result = OneOf([t1], p=0.0)(ep)
    assert t1.call_count == 0
    assert result is ep


def test_oneof_empty_returns_episode_unchanged():
    """OneOf with an empty list returns the episode as-is."""
    ep = _make_episode()
    result = OneOf([], p=1.0)(ep)
    assert result is ep


# ---------------------------------------------------------------------------
# SomeOf tests
# ---------------------------------------------------------------------------

def test_someof_picks_within_range():
    """Over many runs, the number of transforms called is always in [n_min, n_max]."""
    n_min, n_max = 1, 3
    for _ in range(50):
        ep = _make_episode()
        transforms = [MockTransform(str(i)) for i in range(5)]
        SomeOf(transforms, n=(n_min, n_max), p=1.0)(ep)
        total_calls = sum(t.call_count for t in transforms)
        assert n_min <= total_calls <= n_max, (
            f"Expected call count in [{n_min}, {n_max}], got {total_calls}"
        )


def test_someof_p0_returns_unchanged():
    """SomeOf with p=0 calls no transform."""
    ep = _make_episode()
    transforms = [MockTransform(str(i)) for i in range(3)]
    result = SomeOf(transforms, n=(1, 2), p=0.0)(ep)
    assert all(t.call_count == 0 for t in transforms)
    assert result is ep


def test_someof_empty_returns_episode_unchanged():
    """SomeOf with an empty list returns the episode as-is."""
    ep = _make_episode()
    result = SomeOf([], n=(1, 2), p=1.0)(ep)
    assert result is ep


# ---------------------------------------------------------------------------
# Nested composition tests
# ---------------------------------------------------------------------------

def test_nested_compose_containing_oneof():
    """Compose containing a OneOf works correctly end-to-end."""
    for _ in range(20):
        ep = _make_episode()
        inner_a = MockTransform("inner_a")
        inner_b = MockTransform("inner_b")
        outer = MockTransform("outer")

        pipeline = Compose(
            [outer, OneOf([inner_a, inner_b], p=1.0)],
            p=1.0,
        )
        pipeline(ep)

        # The outer transform must have been called once
        assert outer.call_count == 1
        # Exactly one of the inner transforms must have been called
        assert inner_a.call_count + inner_b.call_count == 1


def test_nested_someof_inside_compose():
    """SomeOf nested inside Compose applies the correct number of inner transforms."""
    for _ in range(20):
        ep = _make_episode()
        pre = MockTransform("pre")
        inner = [MockTransform(f"inner_{i}") for i in range(4)]

        pipeline = Compose(
            [pre, SomeOf(inner, n=(2, 3), p=1.0)],
            p=1.0,
        )
        pipeline(ep)

        assert pre.call_count == 1
        inner_calls = sum(t.call_count for t in inner)
        assert 2 <= inner_calls <= 3
