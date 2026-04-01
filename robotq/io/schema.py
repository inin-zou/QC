"""LeRobot v3 schema parsing utilities.

This module handles reading and validating the metadata files that describe a
LeRobot v3 dataset's structure.  It knows nothing about video decoding or
episode objects – pure JSON / JSONL parsing only.
"""

import json
from pathlib import Path

# Keys that must be present in info.json for the file to be considered valid.
_REQUIRED_INFO_KEYS = (
    "codebase_version",
    "fps",
    "total_episodes",
    "features",
    "chunks_size",
    "data_path",
    "video_path",
)


def parse_info(path: str | Path) -> dict:
    """Read and return *info.json* as a dict.

    Parameters
    ----------
    path:
        Filesystem path to ``info.json``.

    Returns
    -------
    dict
        The parsed JSON object.

    Raises
    ------
    ValueError
        If any of the required keys is missing from the file.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        info = json.load(fh)

    missing = [k for k in _REQUIRED_INFO_KEYS if k not in info]
    if missing:
        raise ValueError(f"info.json is missing required key(s): {missing}")

    return info


def parse_tasks(path: str | Path) -> list[dict]:
    """Read *tasks.jsonl* and return a list of task dicts.

    Each line of the file is expected to be a JSON object with at least
    ``task_index`` (int) and ``task`` (str) fields.

    Parameters
    ----------
    path:
        Filesystem path to ``tasks.jsonl``.

    Returns
    -------
    list[dict]
        One dict per non-empty line, in file order.  Returns an empty list
        when the file is empty or contains only whitespace lines.
    """
    path = Path(path)
    tasks: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tasks.append({"task_index": int(obj["task_index"]), "task": str(obj["task"])})
    return tasks


def get_video_path(info: dict, episode_index: int, video_key: str) -> str:
    """Construct the video file path for a specific episode and camera.

    Uses the ``video_path`` template stored in *info*.  The template format
    follows the LeRobot v3 convention, e.g.:
    ``"videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"``

    Parameters
    ----------
    info:
        Parsed info dict (from :func:`parse_info`).
    episode_index:
        Zero-based episode index.
    video_key:
        Camera / sensor key, e.g. ``"cam_high"``.

    Returns
    -------
    str
        Rendered path string.
    """
    episode_chunk = episode_index // info["chunks_size"]
    template: str = info["video_path"]
    return template.format(
        episode_chunk=episode_chunk,
        video_key=video_key,
        episode_index=episode_index,
    )


def get_data_path(info: dict, episode_index: int) -> str:
    """Construct the parquet file path for a specific episode.

    Uses the ``data_path`` template stored in *info*.  The template format
    follows the LeRobot v3 convention, e.g.:
    ``"data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"``

    Parameters
    ----------
    info:
        Parsed info dict (from :func:`parse_info`).
    episode_index:
        Zero-based episode index.

    Returns
    -------
    str
        Rendered path string.
    """
    episode_chunk = episode_index // info["chunks_size"]
    template: str = info["data_path"]
    return template.format(
        episode_chunk=episode_chunk,
        episode_index=episode_index,
    )


def get_camera_names(info: dict) -> list[str]:
    """Extract camera names from the *features* dict in *info*.

    Camera features are identified by keys that start with
    ``"observation.images."``.  This function returns the trailing segment
    after the last ``"."`` for each such key.

    Example
    -------
    A key ``"observation.images.cam_high"`` yields ``"cam_high"``.

    Parameters
    ----------
    info:
        Parsed info dict (from :func:`parse_info`).

    Returns
    -------
    list[str]
        Sorted list of camera name strings (sorted for determinism).
    """
    prefix = "observation.images."
    names = [key[len(prefix) :] for key in info.get("features", {}) if key.startswith(prefix)]
    return sorted(names)
