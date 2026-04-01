"""Episode: canonical data container for the RobotQ pipeline.

Every transform takes an Episode and returns an Episode.
Never pass frames, actions, or states as separate arguments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class EpisodeMetadata:
    """Structured metadata associated with a single episode."""

    episode_index: int
    task_description: str
    task_id: int
    fps: float
    camera_names: list[str]
    robot_type: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Episode:
    """Canonical data container used throughout the RobotQ project.

    Attributes
    ----------
    frames:
        Per-camera image sequences keyed by camera name.
        Each value is a list of H×W×C uint8 arrays of length T.
    actions:
        Action array of shape (T, action_dim).
    states:
        State array of shape (T, state_dim).
    metadata:
        Structured episode metadata.
    """

    frames: dict[str, list[np.ndarray]]
    actions: np.ndarray
    states: np.ndarray
    metadata: EpisodeMetadata

    def __post_init__(self) -> None:
        """Validate internal consistency of frame counts and time dimensions."""
        # Collect per-camera frame counts
        camera_lengths = {cam: len(imgs) for cam, imgs in self.frames.items()}

        # All cameras must have the same number of frames
        unique_lengths = set(camera_lengths.values())
        if len(unique_lengths) > 1:
            detail = ", ".join(f"{c}={n}" for c, n in camera_lengths.items())
            raise ValueError(
                f"Frame counts differ across cameras: {detail}"
            )

        # If there are any frames, validate against actions / states T dimension
        if camera_lengths:
            frames_T = next(iter(unique_lengths))
            actions_T = self.actions.shape[0]
            states_T = self.states.shape[0]

            if frames_T != actions_T:
                raise ValueError(
                    f"Frame count ({frames_T}) does not match actions T dimension ({actions_T})."
                )
            if frames_T != states_T:
                raise ValueError(
                    f"Frame count ({frames_T}) does not match states T dimension ({states_T})."
                )

    # ------------------------------------------------------------------
    # Helper properties
    # ------------------------------------------------------------------

    @property
    def num_frames(self) -> int:
        """Number of timesteps in this episode."""
        return self.actions.shape[0]

    @property
    def action_dim(self) -> int:
        """Dimensionality of the action space."""
        return self.actions.shape[1] if self.actions.ndim > 1 else 1

    @property
    def state_dim(self) -> int:
        """Dimensionality of the state space."""
        return self.states.shape[1] if self.states.ndim > 1 else 1
