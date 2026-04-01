"""Mirror: horizontally flip video + swap left/right arm actions via adapter."""

from __future__ import annotations

import copy

import numpy as np

from robotq.adapters.base import ActionAdapter
from robotq.core.episode import Episode
from robotq.core.transform import RobotTransform


class Mirror(RobotTransform):
    """Horizontally flip all camera frames and swap left/right arm actions.

    This is the key robotics-specific augmentation. It requires an
    ActionAdapter to know how to remap the action/state vectors.

    Steps:
    1. Flip all camera frames horizontally (np.fliplr)
    2. Swap left/right arm actions via adapter.swap_arms_batch()
    3. Swap left/right arm states via adapter.swap_arms_batch()
    4. Apply flip signs via adapter.get_flip_signs()
    """

    def apply_to_episode(self, episode: Episode, adapter: ActionAdapter) -> Episode:
        # 1. Flip all camera frames horizontally
        new_frames: dict[str, list[np.ndarray]] = {}
        for cam, frame_list in episode.frames.items():
            new_frames[cam] = [np.ascontiguousarray(np.fliplr(f)) for f in frame_list]

        # 2. Swap + flip actions
        flip_signs = adapter.get_flip_signs()
        swapped_actions = adapter.swap_arms_batch(episode.actions)
        if flip_signs.shape[0] == episode.actions.shape[1]:
            new_actions = swapped_actions * flip_signs.astype(episode.actions.dtype)
        else:
            # Adapter doesn't define flip signs for this action dim (e.g. generic adapter)
            new_actions = swapped_actions

        # 3. Swap + flip states (only if adapter supports this dim)
        if episode.states.shape[1] == episode.actions.shape[1]:
            swapped_states = adapter.swap_arms_batch(episode.states)
            if flip_signs.shape[0] == episode.states.shape[1]:
                new_states = swapped_states * flip_signs.astype(episode.states.dtype)
            else:
                new_states = swapped_states
        else:
            # State dim differs from action dim — adapter can't handle it
            new_states = episode.states.copy()

        # 4. Build new episode with updated metadata
        new_metadata = copy.deepcopy(episode.metadata)

        return Episode(
            frames=new_frames,
            actions=new_actions,
            states=new_states,
            metadata=new_metadata,
        )
