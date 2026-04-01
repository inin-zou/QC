"""Base protocol definition for robot action adapters."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class ActionAdapter(Protocol):
    """Protocol that all robot action adapters must satisfy.

    An ActionAdapter encapsulates robot-specific knowledge about how to
    interpret and manipulate action vectors (e.g. joint positions for a
    bimanual arm).
    """

    @property
    def robot_type(self) -> str:
        """Unique string identifier for the robot (e.g. 'aloha')."""
        ...

    def get_left_slice(self) -> slice:
        """Return the slice that selects the left-arm DOFs from an action vector."""
        ...

    def get_right_slice(self) -> slice:
        """Return the slice that selects the right-arm DOFs from an action vector."""
        ...

    def get_flip_signs(self) -> np.ndarray:
        """Return a sign array of shape (action_dim,) with values +1 or -1.

        Multiplying an action vector by this array produces the horizontally
        mirrored version of that action (left/right symmetry).
        """
        ...

    def swap_arms(self, vec: np.ndarray) -> np.ndarray:
        """Return a copy of the 1-D action vector with left and right slices swapped."""
        ...

    def swap_arms_batch(self, vecs: np.ndarray) -> np.ndarray:
        """Return a copy of a (T, action_dim) array with left and right slices swapped."""
        ...
