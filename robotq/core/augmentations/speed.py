"""SpeedWarp augmentation: resample an episode at a different playback rate."""

from __future__ import annotations

import copy
import random

import numpy as np

from robotq.core.episode import Episode
from robotq.core.transform import TrajectoryTransform


def _interp_linear(orig: np.ndarray, new_indices: np.ndarray) -> np.ndarray:
    """Linear interpolation without scipy.

    Parameters
    ----------
    orig:
        Source array of shape (T, D).
    new_indices:
        Float indices into the first axis of *orig*.

    Returns
    -------
    np.ndarray
        Interpolated array of shape (len(new_indices), D) with the same dtype
        as *orig*.
    """
    idx_floor = np.floor(new_indices).astype(int)
    idx_ceil = np.minimum(idx_floor + 1, len(orig) - 1)
    frac = (new_indices - idx_floor)[:, None]
    return (orig[idx_floor] * (1 - frac) + orig[idx_ceil] * frac).astype(orig.dtype)


class SpeedWarp(TrajectoryTransform):
    """Resample episode at a randomly chosen speed factor.

    A factor < 1.0 slows down (more frames), factor > 1.0 speeds up (fewer
    frames).

    Parameters
    ----------
    min_rate:
        Minimum speed multiplier (must be > 0).
    max_rate:
        Maximum speed multiplier (must be >= *min_rate*).
    p:
        Probability of applying the transform to any given episode call.
    """

    def __init__(
        self,
        min_rate: float = 0.8,
        max_rate: float = 1.2,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        if min_rate <= 0 or max_rate < min_rate:
            raise ValueError(f"Need 0 < min_rate <= max_rate, got ({min_rate}, {max_rate})")
        self.min_rate = min_rate
        self.max_rate = max_rate

    def apply_to_episode(self, episode: Episode) -> Episode:
        # 1. Sample speed factor
        rate = random.uniform(self.min_rate, self.max_rate)

        # 2. Compute new frame count
        orig_len = episode.num_frames
        new_len = max(1, round(orig_len / rate))

        # 3. Compute source indices (linear interpolation over original range)
        new_indices = np.linspace(0, orig_len - 1, new_len)

        # 4. Resample frames (nearest-neighbor for images)
        nearest_idx = np.round(new_indices).astype(int)
        new_frames = {}
        for cam, frame_list in episode.frames.items():
            new_frames[cam] = [frame_list[i] for i in nearest_idx]

        # 5. Interpolate actions and states (linear, numpy-only)
        new_actions = _interp_linear(episode.actions, new_indices)
        new_states = _interp_linear(episode.states, new_indices)

        # 6. Update metadata
        new_metadata = copy.deepcopy(episode.metadata)

        return Episode(
            frames=new_frames,
            actions=new_actions,
            states=new_states,
            metadata=new_metadata,
        )
