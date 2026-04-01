"""Tests for transform base classes."""

import numpy as np

from robotq.core.episode import Episode, EpisodeMetadata
from robotq.core.transform import FrameTransform, SequenceTransform, TrajectoryTransform


def _make_episode(n_frames=5):
    return Episode(
        frames={"cam": [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(n_frames)]},
        actions=np.zeros((n_frames, 2), dtype=np.float32),
        states=np.zeros((n_frames, 2), dtype=np.float32),
        metadata=EpisodeMetadata(
            episode_index=0, task_description="test", task_id=0,
            fps=50.0, camera_names=["cam"], robot_type="test",
        ),
    )


class ConcreteFrameTransform(FrameTransform):
    def __init__(self, p=1.0):
        super().__init__(p=p)
        self.call_count = 0

    def apply_to_frame(self, frame):
        self.call_count += 1
        return frame + 1


class ConcreteSequenceTransform(SequenceTransform):
    def __init__(self, p=1.0):
        super().__init__(p=p)
        self.params_calls = 0

    def get_params(self, episode):
        self.params_calls += 1
        return {"val": 42}

    def apply_to_frame(self, frame, params):
        return frame + params["val"]


class ConcreteTrajectoryTransform(TrajectoryTransform):
    def __init__(self, p=1.0):
        super().__init__(p=p)
        self.called = False

    def apply_to_episode(self, episode):
        self.called = True
        return episode


def test_frame_transform_maps_all_frames():
    ep = _make_episode(5)
    t = ConcreteFrameTransform(p=1.0)
    result = t(ep)
    assert t.call_count == 5
    assert result.frames["cam"][0][0, 0, 0] == 1


def test_frame_transform_p0_unchanged():
    ep = _make_episode(5)
    t = ConcreteFrameTransform(p=0.0)
    result = t(ep)
    assert t.call_count == 0
    assert result is ep


def test_sequence_transform_params_called_once():
    ep = _make_episode(5)
    t = ConcreteSequenceTransform(p=1.0)
    t(ep)
    assert t.params_calls == 1


def test_sequence_transform_same_params_all_frames():
    ep = _make_episode(3)
    t = ConcreteSequenceTransform(p=1.0)
    result = t(ep)
    # All frames should have value 42 (0 + 42)
    for frame in result.frames["cam"]:
        assert frame[0, 0, 0] == 42


def test_trajectory_transform_delegates():
    ep = _make_episode(5)
    t = ConcreteTrajectoryTransform(p=1.0)
    t(ep)
    assert t.called


def test_trajectory_transform_p0_unchanged():
    ep = _make_episode(5)
    t = ConcreteTrajectoryTransform(p=0.0)
    result = t(ep)
    assert not t.called
    assert result is ep


def test_transform_repr():
    t = ConcreteFrameTransform(p=0.5)
    assert "ConcreteFrameTransform" in repr(t)
    assert "p=0.5" in repr(t)
