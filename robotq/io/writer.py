"""Dataset writer that creates LeRobot v3 datasets from Episode objects.

This module uses LeRobot's official API (LeRobotDataset.create, add_frame,
save_episode, finalize) to guarantee that the output is a valid LeRobot v3
dataset compatible with the HF visualizer and training tools.

We intentionally avoid writing raw parquet / video files ourselves.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from robotq.core.episode import Episode

logger = logging.getLogger(__name__)


def write_dataset(
    episodes: list[Episode],
    *,
    repo_id: str,
    root: str | Path | None = None,
    local_only: bool = False,
    token: str | None = None,
) -> str:
    """Write a list of Episode objects into a new LeRobot v3 dataset.

    Parameters
    ----------
    episodes:
        Non-empty list of :class:`Episode` objects.  All episodes must share
        the same camera names, action dimension, state dimension, and fps.
    repo_id:
        HuggingFace-style ``"user/dataset"`` identifier.
    root:
        Local directory for the dataset.  If *None*, LeRobot uses its
        default cache location (``HF_LEROBOT_HOME / repo_id``).
    local_only:
        When *True* the dataset is written to disk but **not** pushed to
        the Hub.
    token:
        HuggingFace API token.  Only used when ``local_only=False``.

    Returns
    -------
    str
        URL to the HuggingFace dataset visualizer.

    Raises
    ------
    ValueError
        If *episodes* is empty.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if not episodes:
        raise ValueError("episodes list must not be empty")

    first = episodes[0]

    # ------------------------------------------------------------------
    # Build features dict from the first episode's structure
    # ------------------------------------------------------------------
    features: dict[str, dict] = {}

    # Camera / video features
    for cam_name in first.metadata.camera_names:
        # Derive spatial shape from the first frame
        sample_frame = first.frames[cam_name][0]  # (H, W, C) uint8
        h, w, c = sample_frame.shape
        features[f"observation.images.{cam_name}"] = {
            "dtype": "video",
            "shape": (h, w, c),
            "names": ["height", "width", "channels"],
        }

    # State feature
    features["observation.state"] = {
        "dtype": "float32",
        "shape": (first.state_dim,),
        "names": None,
    }

    # Action feature
    features["action"] = {
        "dtype": "float32",
        "shape": (first.action_dim,),
        "names": None,
    }

    # ------------------------------------------------------------------
    # Create the dataset via LeRobot's official API
    # ------------------------------------------------------------------
    create_kwargs: dict = dict(
        repo_id=repo_id,
        fps=int(first.metadata.fps),
        features=features,
        robot_type=first.metadata.robot_type,
    )
    if root is not None:
        create_kwargs["root"] = root

    dataset = LeRobotDataset.create(**create_kwargs)

    # ------------------------------------------------------------------
    # Populate frames
    # ------------------------------------------------------------------
    for episode in episodes:
        for t in range(episode.num_frames):
            frame: dict = {}

            # Task description — required by add_frame() for every frame
            frame["task"] = episode.metadata.task_description

            # Camera images: numpy (H, W, C) uint8
            for cam_name in episode.metadata.camera_names:
                frame[f"observation.images.{cam_name}"] = episode.frames[cam_name][t]

            # State: numpy float32 with shape (state_dim,)
            frame["observation.state"] = episode.states[t].astype(np.float32)

            # Action: numpy float32 with shape (action_dim,)
            frame["action"] = episode.actions[t].astype(np.float32)

            dataset.add_frame(frame)

        # Save the completed episode
        dataset.save_episode()

    # ------------------------------------------------------------------
    # Finalize — flush parquet footers (CRITICAL)
    # ------------------------------------------------------------------
    dataset.finalize()

    # Verify episode count matches expectation
    written_count = dataset.meta.total_episodes
    if written_count != len(episodes):
        raise RuntimeError(
            f"Writer integrity check failed: expected {len(episodes)} episodes, "
            f"finalize() produced {written_count}."
        )

    # ------------------------------------------------------------------
    # Optionally push to Hub
    # ------------------------------------------------------------------
    if not local_only:
        # LeRobot's push_to_hub uses HfApi() which reads the token from
        # the environment (HF_TOKEN) or from ``huggingface-cli login``.
        # If a token was explicitly provided, inject it into the env so
        # that HfApi picks it up.
        if token is not None:
            import os

            os.environ["HF_TOKEN"] = token
        dataset.meta.push_to_hub()

    return generate_visualizer_link(repo_id)


def generate_visualizer_link(repo_id: str, episode: int = 0) -> str:
    """Build a URL pointing to the HuggingFace LeRobot dataset visualizer.

    Parameters
    ----------
    repo_id:
        HuggingFace ``"user/dataset"`` identifier.
    episode:
        Zero-based episode index to visualize.

    Returns
    -------
    str
        Fully-encoded URL.
    """
    encoded_id = repo_id.replace("/", "%2F")
    return (
        f"https://huggingface.co/spaces/lerobot/visualize_dataset"
        f"?path=%2F{encoded_id}%2Fepisode_{episode}"
    )
