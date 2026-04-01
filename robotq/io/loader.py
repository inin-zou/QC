"""Dataset loader for LeRobot v3 datasets.

Uses LeRobot's API for metadata and tabular data, and OpenCV for video
decoding (bypasses torchcodec dependency issues).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from robotq.core.episode import Episode, EpisodeMetadata
from robotq.io.video import decode_video

logger = logging.getLogger(__name__)


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
        HuggingFace Hub dataset ID (e.g. ``"lerobot/aloha_static_cups_open"``).
    max_episodes:
        Maximum number of episodes to load. None means load all.
    local_dir:
        If provided, read from this local directory instead of downloading.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # Load dataset via LeRobot API (downloads if needed, parses metadata)
    kwargs: dict = {"repo_id": repo_id}
    if local_dir is not None:
        kwargs["root"] = local_dir
    dataset = LeRobotDataset(**kwargs)

    # Extract metadata
    fps = float(dataset.fps)
    robot_type = dataset.meta.robot_type or "unknown"
    total_episodes = dataset.meta.total_episodes

    # Camera names from features
    camera_names = sorted(
        key.removeprefix("observation.images.")
        for key in dataset.features
        if key.startswith("observation.images.")
    )

    # Task lookup from meta
    task_lookup: dict[int, str] = {}
    if hasattr(dataset.meta, "tasks") and dataset.meta.tasks is not None:
        tasks_df = dataset.meta.tasks
        for task_desc, row in tasks_df.iterrows():
            task_lookup[int(row["task_index"])] = str(task_desc)

    # Dataset root on disk
    dataset_root = Path(dataset.root)

    # Validate dataset is not empty
    if total_episodes == 0:
        raise ValueError(f"Dataset '{repo_id}' contains 0 episodes.")

    # Determine how many episodes to load
    num_episodes = total_episodes
    if max_episodes is not None:
        num_episodes = min(max_episodes, total_episodes)

    # Read episode metadata to get boundaries and file paths
    episodes_meta = dataset.meta.episodes
    # hf_dataset has all tabular data
    hf_ds = dataset.hf_dataset

    # Warn about memory usage for large loads
    if num_episodes > 10 and len(camera_names) >= 4:
        logger.warning(
            "Loading %d episodes with %d cameras — estimated ~%.1fGB RAM. "
            "Use max_episodes to limit if needed.",
            num_episodes,
            len(camera_names),
            num_episodes * len(camera_names) * 400 * 480 * 640 * 3 / 1e9,
        )

    episodes: list[Episode] = []

    for ep_idx in range(num_episodes):
        ep_row = episodes_meta[ep_idx]

        # Get frame range from episode metadata
        from_idx = int(ep_row["dataset_from_index"])
        to_idx = int(ep_row["dataset_to_index"])
        n_frames = to_idx - from_idx

        if n_frames <= 0:
            raise ValueError(
                f"Episode {ep_idx} has invalid frame range [{from_idx}, {to_idx}) "
                f"in dataset '{repo_id}'."
            )

        # Extract actions and states from hf_dataset
        ep_slice = hf_ds.select(range(from_idx, to_idx))
        actions = np.array(ep_slice["action"], dtype=np.float32)
        states = np.array(ep_slice["observation.state"], dtype=np.float32)

        # Task info
        task_indices = ep_slice["task_index"]
        task_id = int(task_indices[0])
        task_description = task_lookup.get(task_id, "")

        # Decode video for each camera using OpenCV (parallel across cameras)
        from concurrent.futures import ThreadPoolExecutor

        def _decode_camera(cam_name: str) -> tuple[str, list[np.ndarray]]:
            video_key = f"observation.images.{cam_name}"
            chunk_idx = int(ep_row.get(f"videos/{video_key}/chunk_index", 0))
            file_idx = int(ep_row.get(f"videos/{video_key}/file_index", 0))
            video_path = (
                dataset_root
                / "videos"
                / video_key
                / f"chunk-{chunk_idx:03d}"
                / f"file-{file_idx:03d}.mp4"
            )
            from_ts = float(ep_row.get(f"videos/{video_key}/from_timestamp", 0.0))
            to_ts = float(ep_row.get(f"videos/{video_key}/to_timestamp", 0.0))
            start_frame = round(from_ts * fps)
            end_frame = round(to_ts * fps)
            return cam_name, decode_video(video_path, start_frame=start_frame, end_frame=end_frame)

        frames: dict[str, list[np.ndarray]] = {}
        with ThreadPoolExecutor(max_workers=len(camera_names)) as pool:
            results = pool.map(_decode_camera, camera_names)
        for cam_name, ep_frames in results:
            # Validate frame count matches tabular data
            if len(ep_frames) != n_frames:
                logger.warning(
                    "Episode %d camera '%s': decoded %d frames but expected %d. "
                    "Trimming/padding to match.",
                    ep_idx,
                    cam_name,
                    len(ep_frames),
                    n_frames,
                )
                if len(ep_frames) > n_frames:
                    ep_frames = ep_frames[:n_frames]
                elif len(ep_frames) == 0:
                    raise ValueError(
                        f"Episode {ep_idx} camera '{cam_name}': decoded 0 frames. "
                        f"Cannot pad from empty."
                    )
                else:
                    while len(ep_frames) < n_frames:
                        ep_frames.append(ep_frames[-1].copy())

            frames[cam_name] = ep_frames

        metadata = EpisodeMetadata(
            episode_index=ep_idx,
            task_description=task_description,
            task_id=task_id,
            fps=fps,
            camera_names=camera_names,
            robot_type=robot_type,
        )

        episodes.append(Episode(frames=frames, actions=actions, states=states, metadata=metadata))

    return episodes
