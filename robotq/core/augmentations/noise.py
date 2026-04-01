"""GaussianNoise and ActionNoise augmentations."""

from __future__ import annotations

import copy

import numpy as np

from robotq.core.episode import Episode
from robotq.core.transform import FrameTransform, TrajectoryTransform


class GaussianNoise(FrameTransform):
    """Add independent per-frame Gaussian noise to pixel values.

    Each frame gets a freshly sampled noise array, so frames within the same
    episode will differ (FrameTransform behaviour).

    Parameters
    ----------
    sigma:
        Noise standard deviation expressed as a fraction of 255.
        The actual noise applied is N(0, sigma * 255).
    p:
        Probability of applying the transform to any given episode call.
    """

    def __init__(self, sigma: float = 0.02, p: float = 1.0) -> None:
        super().__init__(p=p)
        self.sigma = sigma

    def apply_to_frame(self, frame: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0.0, self.sigma * 255, frame.shape)
        noisy = frame.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)


class ActionNoise(TrajectoryTransform):
    """Add Gaussian noise to the action trajectory of an episode.

    Frames and states are left unchanged.  A new Episode is always returned;
    the input episode is never mutated.

    Parameters
    ----------
    sigma:
        Standard deviation of the noise added to each action component.
        The noise is drawn from N(0, sigma).
    p:
        Probability of applying the transform to any given episode call.
    """

    def __init__(self, sigma: float = 0.01, p: float = 1.0) -> None:
        super().__init__(p=p)
        self.sigma = sigma

    def apply_to_episode(self, episode: Episode) -> Episode:
        noise = np.random.normal(0.0, self.sigma, episode.actions.shape).astype(
            episode.actions.dtype
        )
        new_actions = episode.actions + noise
        return Episode(
            frames=episode.frames,
            actions=new_actions,
            states=episode.states,
            metadata=copy.deepcopy(episode.metadata),
        )
