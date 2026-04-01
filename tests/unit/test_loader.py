"""Unit tests for robotq.io.loader."""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import polars as pl
import pytest

from robotq.io.loader import load_dataset


# ---------------------------------------------------------------------------
# Helpers for building mock dataset fixtures
# ---------------------------------------------------------------------------

def _make_info(
    *,
    total_episodes: int = 1,
    fps: int = 30,
    robot_type: str = "test_robot",
    chunks_size: int = 1000,
    camera_names: list[str] | None = None,
) -> dict:
    """Build a minimal info.json dict."""
    if camera_names is None:
        camera_names = ["cam_high"]
    features: dict = {
        "action": {"dtype": "float32", "shape": [6], "names": None},
        "observation.state": {"dtype": "float32", "shape": [6], "names": None},
    }
    for cam in camera_names:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": [64, 64, 3],
            "names": None,
            "video_info": {"video.fps": fps, "video.codec": "av1"},
        }
    return {
        "codebase_version": "v3.0",
        "fps": fps,
        "total_episodes": total_episodes,
        "features": features,
        "robot_type": robot_type,
        "chunks_size": chunks_size,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    }


def _make_tasks(descriptions: list[str]) -> list[dict]:
    """Build a list of task dicts for tasks.jsonl."""
    return [{"task_index": i, "task": desc} for i, desc in enumerate(descriptions)]


def _write_tasks_jsonl(path: Path, tasks: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for t in tasks:
            fh.write(json.dumps(t) + "\n")


def _make_parquet(
    path: Path,
    *,
    episode_index: int,
    num_frames: int,
    action_dim: int,
    state_dim: int,
    task_index: int = 0,
) -> None:
    """Write a minimal LeRobot v3 parquet file."""
    rng = np.random.default_rng(42 + episode_index)
    actions = rng.standard_normal((num_frames, action_dim)).astype(np.float32)
    states = rng.standard_normal((num_frames, state_dim)).astype(np.float32)

    df = pl.DataFrame({
        "episode_index": pl.Series([episode_index] * num_frames, dtype=pl.Int64),
        "frame_index": pl.Series(list(range(num_frames)), dtype=pl.Int64),
        "timestamp": pl.Series([i / 30.0 for i in range(num_frames)], dtype=pl.Float64),
        "index": pl.Series(list(range(num_frames)), dtype=pl.Int64),
        "task_index": pl.Series([task_index] * num_frames, dtype=pl.Int64),
        "next.done": pl.Series(
            [False] * (num_frames - 1) + [True], dtype=pl.Boolean
        ),
        "action": pl.Series([row.tolist() for row in actions]),
        "observation.state": pl.Series([row.tolist() for row in states]),
    })

    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def _make_video(path: Path, *, num_frames: int, width: int = 64, height: int = 64) -> None:
    """Create a short MP4 video with solid-color frames using OpenCV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (width, height))
    rng = np.random.default_rng(123)
    for _ in range(num_frames):
        # Random solid color per frame for easy verification
        color = rng.integers(0, 256, size=3).tolist()
        frame = np.full((height, width, 3), color, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _build_mock_dataset(
    root: Path,
    *,
    num_episodes: int = 1,
    num_frames: int = 5,
    action_dim: int = 6,
    state_dim: int = 6,
    camera_names: list[str] | None = None,
    task_descriptions: list[str] | None = None,
    fps: int = 30,
    robot_type: str = "test_robot",
) -> None:
    """Build a complete mock LeRobot v3 dataset directory."""
    if camera_names is None:
        camera_names = ["cam_high"]
    if task_descriptions is None:
        task_descriptions = ["pick up the cup"]

    info = _make_info(
        total_episodes=num_episodes,
        fps=fps,
        robot_type=robot_type,
        camera_names=camera_names,
    )

    # Write meta/info.json
    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "info.json").write_text(json.dumps(info), encoding="utf-8")

    # Write meta/tasks.jsonl
    tasks = _make_tasks(task_descriptions)
    _write_tasks_jsonl(meta_dir / "tasks.jsonl", tasks)

    # Write parquet and video files for each episode
    for ep_idx in range(num_episodes):
        episode_chunk = ep_idx // info["chunks_size"]

        # Parquet
        parquet_path = root / f"data/chunk-{episode_chunk:03d}/episode_{ep_idx:06d}.parquet"
        _make_parquet(
            parquet_path,
            episode_index=ep_idx,
            num_frames=num_frames,
            action_dim=action_dim,
            state_dim=state_dim,
            task_index=0,
        )

        # Video per camera
        for cam_name in camera_names:
            video_key = f"observation.images.{cam_name}"
            video_path = (
                root
                / f"videos/chunk-{episode_chunk:03d}/{video_key}/episode_{ep_idx:06d}.mp4"
            )
            _make_video(video_path, num_frames=num_frames)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_dataset(tmp_path: Path) -> Path:
    """Single-episode mock dataset with 5 frames."""
    root = tmp_path / "mock_dataset"
    _build_mock_dataset(root, num_episodes=1, num_frames=5, action_dim=6, state_dim=6)
    return root


@pytest.fixture()
def multi_episode_dataset(tmp_path: Path) -> Path:
    """Two-episode mock dataset for max_episodes testing."""
    root = tmp_path / "multi_ep"
    _build_mock_dataset(root, num_episodes=2, num_frames=5, action_dim=6, state_dim=6)
    return root


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadDataset:
    """Tests for load_dataset()."""

    def test_frame_count(self, mock_dataset: Path) -> None:
        """Loaded episode should have the correct number of frames."""
        episodes = load_dataset("unused/repo", local_dir=str(mock_dataset))
        assert len(episodes) == 1
        assert episodes[0].num_frames == 5

    def test_action_dim(self, mock_dataset: Path) -> None:
        """Loaded episode should have the correct action dimensionality."""
        episodes = load_dataset("unused/repo", local_dir=str(mock_dataset))
        assert episodes[0].action_dim == 6

    def test_state_dim(self, mock_dataset: Path) -> None:
        """Loaded episode should have the correct state dimensionality."""
        episodes = load_dataset("unused/repo", local_dir=str(mock_dataset))
        assert episodes[0].state_dim == 6

    def test_metadata_fps(self, mock_dataset: Path) -> None:
        """Metadata fps should match info.json."""
        episodes = load_dataset("unused/repo", local_dir=str(mock_dataset))
        assert episodes[0].metadata.fps == 30.0

    def test_metadata_camera_names(self, mock_dataset: Path) -> None:
        """Metadata camera_names should match cameras in info.json features."""
        episodes = load_dataset("unused/repo", local_dir=str(mock_dataset))
        assert episodes[0].metadata.camera_names == ["cam_high"]

    def test_metadata_robot_type(self, mock_dataset: Path) -> None:
        """Metadata robot_type should match info.json."""
        episodes = load_dataset("unused/repo", local_dir=str(mock_dataset))
        assert episodes[0].metadata.robot_type == "test_robot"

    def test_metadata_episode_index(self, mock_dataset: Path) -> None:
        """Metadata episode_index should be 0 for the first episode."""
        episodes = load_dataset("unused/repo", local_dir=str(mock_dataset))
        assert episodes[0].metadata.episode_index == 0

    def test_metadata_task_description(self, mock_dataset: Path) -> None:
        """Metadata task_description should come from tasks.jsonl."""
        episodes = load_dataset("unused/repo", local_dir=str(mock_dataset))
        assert episodes[0].metadata.task_description == "pick up the cup"

    def test_metadata_task_id(self, mock_dataset: Path) -> None:
        """Metadata task_id should match the task_index from the parquet data."""
        episodes = load_dataset("unused/repo", local_dir=str(mock_dataset))
        assert episodes[0].metadata.task_id == 0

    def test_frames_dict_keys(self, mock_dataset: Path) -> None:
        """Episode frames dict should be keyed by camera name."""
        episodes = load_dataset("unused/repo", local_dir=str(mock_dataset))
        assert list(episodes[0].frames.keys()) == ["cam_high"]

    def test_frames_content(self, mock_dataset: Path) -> None:
        """Each frame should be a uint8 H x W x C numpy array."""
        episodes = load_dataset("unused/repo", local_dir=str(mock_dataset))
        frame = episodes[0].frames["cam_high"][0]
        assert isinstance(frame, np.ndarray)
        assert frame.dtype == np.uint8
        assert frame.shape == (64, 64, 3)

    def test_max_episodes_limits_output(self, multi_episode_dataset: Path) -> None:
        """max_episodes=1 should return only one episode even if dataset has two."""
        episodes = load_dataset(
            "unused/repo",
            local_dir=str(multi_episode_dataset),
            max_episodes=1,
        )
        assert len(episodes) == 1
        assert episodes[0].metadata.episode_index == 0

    def test_all_episodes_loaded_without_limit(self, multi_episode_dataset: Path) -> None:
        """Without max_episodes, all episodes should be loaded."""
        episodes = load_dataset("unused/repo", local_dir=str(multi_episode_dataset))
        assert len(episodes) == 2
        assert episodes[0].metadata.episode_index == 0
        assert episodes[1].metadata.episode_index == 1

    def test_actions_are_float32(self, mock_dataset: Path) -> None:
        """Actions array should be float32."""
        episodes = load_dataset("unused/repo", local_dir=str(mock_dataset))
        assert episodes[0].actions.dtype == np.float32

    def test_states_are_float32(self, mock_dataset: Path) -> None:
        """States array should be float32."""
        episodes = load_dataset("unused/repo", local_dir=str(mock_dataset))
        assert episodes[0].states.dtype == np.float32
