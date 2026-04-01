"""Dataset loader for LeRobot v3 datasets.

Downloads (or reads locally) a LeRobot v3 dataset and converts each episode
into the canonical :class:`~robotq.core.episode.Episode` container.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import polars as pl
from huggingface_hub import snapshot_download

from robotq.core.episode import Episode, EpisodeMetadata
from robotq.io import schema, video


def load_dataset(
    repo_id: str,
    *,
    max_episodes: int | None = None,
    local_dir: str | None = None,
) -> list[Episode]:
    """Load a LeRobot v3 dataset and return a list of Episode objects.

    Parameters
    ----------
    repo_id:
        HuggingFace Hub repository ID (e.g. ``"lerobot/aloha_static_cups_open"``).
        Used for ``snapshot_download`` when *local_dir* is not provided.
    max_episodes:
        Maximum number of episodes to load.  ``None`` means load all episodes
        present in the dataset.
    local_dir:
        If provided, read the dataset from this local directory instead of
        downloading from the Hub.

    Returns
    -------
    list[Episode]
        One :class:`Episode` per loaded episode, ordered by episode index.
    """
    # ------------------------------------------------------------------
    # 1. Resolve dataset root
    # ------------------------------------------------------------------
    if local_dir is not None:
        dataset_root = Path(local_dir)
    else:
        dataset_root = Path(snapshot_download(repo_id))

    # ------------------------------------------------------------------
    # 2. Parse metadata
    # ------------------------------------------------------------------
    info = schema.parse_info(dataset_root / "meta" / "info.json")
    tasks = schema.parse_tasks(dataset_root / "meta" / "tasks.jsonl")

    fps: float = float(info["fps"])
    robot_type: str = info.get("robot_type", "unknown")
    camera_names: list[str] = schema.get_camera_names(info)
    total_episodes: int = info["total_episodes"]

    num_episodes = total_episodes
    if max_episodes is not None:
        num_episodes = min(max_episodes, total_episodes)

    # Build a task_index -> task description lookup
    task_lookup: dict[int, str] = {t["task_index"]: t["task"] for t in tasks}

    # ------------------------------------------------------------------
    # 3. Load each episode
    # ------------------------------------------------------------------
    episodes: list[Episode] = []

    for ep_idx in range(num_episodes):
        # 3a. Read parquet
        data_path = dataset_root / schema.get_data_path(info, ep_idx)
        df = pl.read_parquet(data_path)

        # 3b. Filter to this episode
        ep_df = df.filter(pl.col("episode_index") == ep_idx)

        # 3c. Extract actions
        actions = np.array(ep_df.get_column("action").to_list(), dtype=np.float32)

        # 3d. Extract states
        states = np.array(
            ep_df.get_column("observation.state").to_list(), dtype=np.float32
        )

        # 3e. Decode video for each camera
        frames: dict[str, list[np.ndarray]] = {}
        for cam_name in camera_names:
            video_key = f"observation.images.{cam_name}"
            vid_rel_path = schema.get_video_path(info, ep_idx, video_key)
            vid_abs_path = os.path.join(dataset_root, vid_rel_path)
            frames[cam_name] = video.decode_video(vid_abs_path)

        # 3f. Look up task info
        task_indices = ep_df.get_column("task_index").to_list()
        # Use the first row's task_index as representative for the episode
        task_id = int(task_indices[0]) if task_indices else 0
        task_description = task_lookup.get(task_id, "")

        # 3g. Build Episode
        metadata = EpisodeMetadata(
            episode_index=ep_idx,
            task_description=task_description,
            task_id=task_id,
            fps=fps,
            camera_names=camera_names,
            robot_type=robot_type,
        )

        episodes.append(
            Episode(
                frames=frames,
                actions=actions,
                states=states,
                metadata=metadata,
            )
        )

    return episodes
