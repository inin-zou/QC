"""Generic single-arm adapter — no arm swap, video-only Mirror.

For robots without bimanual (left/right) arm symmetry. Mirror will
flip video frames but leave actions and states unchanged.
"""

from __future__ import annotations

import numpy as np


class GenericSingleArmAdapter:
    """Adapter for single-arm or non-bimanual robots.

    Mirror with this adapter flips video only — actions and states
    are copied unchanged since there's no left/right arm symmetry.
    """

    @property
    def robot_type(self) -> str:
        return "generic"

    def get_left_slice(self) -> slice:
        return slice(0, 0)  # Empty — no left arm

    def get_right_slice(self) -> slice:
        return slice(0, 0)  # Empty — no right arm

    def get_flip_signs(self) -> np.ndarray:
        # Return empty — mirror.py checks state_dim != action_dim and copies
        return np.array([], dtype=np.float64)

    def swap_arms(self, vec: np.ndarray) -> np.ndarray:
        return vec.copy()  # No swap

    def swap_arms_batch(self, vecs: np.ndarray) -> np.ndarray:
        return vecs.copy()  # No swap
