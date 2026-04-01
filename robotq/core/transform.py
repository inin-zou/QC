"""Transform base classes for the RobotQ augmentation pipeline.

Four base classes, each for a different augmentation pattern:
- FrameTransform: per-frame independent (different random params per frame)
- SequenceTransform: temporally consistent (same params across episode)
- TrajectoryTransform: episode-level (may change length/fps/metadata)
- RobotTransform: paired image+action (requires ActionAdapter)
"""

from __future__ import annotations

import copy
import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from robotq.core.episode import Episode

if TYPE_CHECKING:
    from robotq.adapters.base import ActionAdapter


class Transform(ABC):
    """Abstract base for all augmentation transforms."""

    def __init__(self, p: float = 1.0) -> None:
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p must be in [0.0, 1.0], got {p!r}")
        self.p = p

    @abstractmethod
    def apply(self, episode: Episode) -> Episode: ...

    def __call__(self, episode: Episode) -> Episode:
        return self.apply(episode)

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in vars(self).items() if not k.startswith("_"))
        return f"{self.__class__.__name__}({params})"


class FrameTransform(Transform):
    """Per-frame independent transform. Each frame gets different random params."""

    @abstractmethod
    def apply_to_frame(self, frame: np.ndarray) -> np.ndarray: ...

    def apply(self, episode: Episode) -> Episode:
        if random.random() > self.p:
            return episode
        new_frames = {}
        for cam, frame_list in episode.frames.items():
            new_frames[cam] = [self.apply_to_frame(f) for f in frame_list]
        return Episode(
            frames=new_frames,
            actions=episode.actions,
            states=episode.states,
            metadata=copy.deepcopy(episode.metadata),
        )


class SequenceTransform(Transform):
    """Temporally consistent transform. Params sampled ONCE per episode."""

    @abstractmethod
    def get_params(self, episode: Episode) -> dict: ...

    @abstractmethod
    def apply_to_frame(self, frame: np.ndarray, params: dict) -> np.ndarray: ...

    def apply(self, episode: Episode) -> Episode:
        if random.random() > self.p:
            return episode
        params = self.get_params(episode)
        new_frames = {}
        for cam, frame_list in episode.frames.items():
            new_frames[cam] = [self.apply_to_frame(f, params) for f in frame_list]
        return Episode(
            frames=new_frames,
            actions=episode.actions,
            states=episode.states,
            metadata=copy.deepcopy(episode.metadata),
        )


class TrajectoryTransform(Transform):
    """Episode-level transform. May change episode length, FPS, or metadata."""

    @abstractmethod
    def apply_to_episode(self, episode: Episode) -> Episode: ...

    def apply(self, episode: Episode) -> Episode:
        if random.random() > self.p:
            return episode
        return self.apply_to_episode(episode)


class RobotTransform(Transform):
    """Paired image+action transform. Requires an ActionAdapter."""

    def __init__(self, adapter: ActionAdapter, p: float = 1.0) -> None:
        super().__init__(p=p)
        self.adapter = adapter

    @abstractmethod
    def apply_to_episode(self, episode: Episode, adapter: ActionAdapter) -> Episode: ...

    def apply(self, episode: Episode) -> Episode:
        if random.random() > self.p:
            return episode
        return self.apply_to_episode(episode, self.adapter)
