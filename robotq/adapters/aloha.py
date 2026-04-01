"""ActionAdapter implementation for the ALOHA bimanual robot.

ALOHA has 14 DOF arranged as two mirrored 7-DOF arms:

  Left arm  (indices 0-6):  waist, shoulder, elbow, forearm_roll,
                             wrist_angle, wrist_rotate, gripper
  Right arm (indices 7-13): same joints in the same order

Reference: https://tonyzhaozh.github.io/aloha/
"""

from __future__ import annotations

import numpy as np

from robotq.adapters.base import ActionAdapter  # noqa: F401 — re-export for convenience

_ACTION_DIM = 14
_LEFT = slice(0, 7)
_RIGHT = slice(7, 14)


class AlohaAdapter:
    """Adapter that encodes ALOHA-specific action-space conventions."""

    # --- Protocol property ---------------------------------------------------

    @property
    def robot_type(self) -> str:
        return "aloha"

    # --- Slice accessors -----------------------------------------------------

    def get_left_slice(self) -> slice:
        """Return slice(0, 7) — the left-arm DOFs."""
        return _LEFT

    def get_right_slice(self) -> slice:
        """Return slice(7, 14) — the right-arm DOFs."""
        return _RIGHT

    # --- Mirror / flip -------------------------------------------------------

    def get_flip_signs(self) -> np.ndarray:
        """Return a sign vector for horizontal (left-right) mirroring.

        For ALOHA the waist joint (index 0 for the left arm, index 7 for the
        right arm) rotates in the *same* world direction for both arms, so
        mirroring requires negating it.  All other joints are negated by the
        arm-swap step itself; their individual signs remain +1.

        NOTE: These signs are an approximation based on the nominal kinematic
        symmetry of the robot.  Real-world calibration may require additional
        per-joint tuning, especially for forearm_roll and wrist_rotate.
        """
        signs = np.ones(_ACTION_DIM, dtype=np.float64)
        signs[0] = -1.0   # left  waist
        signs[7] = -1.0   # right waist
        return signs

    # --- Arm-swap helpers ----------------------------------------------------

    def swap_arms(self, vec: np.ndarray) -> np.ndarray:
        """Return a copy of the 1-D action vector with left/right arms swapped.

        Parameters
        ----------
        vec:
            1-D array of shape (14,).

        Returns
        -------
        np.ndarray
            New array of shape (14,) where left and right halves are exchanged.
        """
        if vec.ndim != 1 or vec.shape[0] != _ACTION_DIM:
            raise ValueError(
                f"swap_arms expects a 1-D array of length {_ACTION_DIM}, "
                f"got shape {vec.shape}"
            )
        out = np.empty_like(vec)
        out[_LEFT] = vec[_RIGHT]
        out[_RIGHT] = vec[_LEFT]
        return out

    def swap_arms_batch(self, vecs: np.ndarray) -> np.ndarray:
        """Return a copy of a (T, 14) array with left/right arms swapped per row.

        Parameters
        ----------
        vecs:
            2-D array of shape (T, 14).

        Returns
        -------
        np.ndarray
            New array of shape (T, 14) where left and right halves are exchanged
            along axis 1 for every time-step.
        """
        if vecs.ndim != 2 or vecs.shape[1] != _ACTION_DIM:
            raise ValueError(
                f"swap_arms_batch expects a 2-D array of shape (T, {_ACTION_DIM}), "
                f"got shape {vecs.shape}"
            )
        out = np.empty_like(vecs)
        out[:, _LEFT] = vecs[:, _RIGHT]
        out[:, _RIGHT] = vecs[:, _LEFT]
        return out
