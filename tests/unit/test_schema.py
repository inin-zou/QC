"""Unit tests for robotq.io.schema."""

import json
import pytest

from robotq.io.schema import (
    get_camera_names,
    get_data_path,
    get_video_path,
    parse_info,
    parse_tasks,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

MINIMAL_INFO = {
    "codebase_version": "v3.0",
    "robot_type": "so100",
    "fps": 30,
    "total_episodes": 50,
    "total_frames": 15000,
    "chunks_size": 1000,
    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    "splits": {"train": "0:50"},
    "features": {
        "observation.images.cam_high": {"dtype": "video", "shape": [480, 640, 3]},
        "observation.images.cam_low": {"dtype": "video", "shape": [480, 640, 3]},
        "action": {"dtype": "float32", "shape": [6]},
        "timestamp": {"dtype": "float32", "shape": [1]},
    },
}


@pytest.fixture()
def info_file(tmp_path):
    """Write MINIMAL_INFO to a temporary info.json and return its path."""
    p = tmp_path / "info.json"
    p.write_text(json.dumps(MINIMAL_INFO), encoding="utf-8")
    return p


@pytest.fixture()
def tasks_file(tmp_path):
    """Write two tasks to a temporary tasks.jsonl and return its path."""
    lines = [
        json.dumps({"task_index": 0, "task": "Pick up the cube"}),
        json.dumps({"task_index": 1, "task": "Place the cube in the bin"}),
    ]
    p = tmp_path / "tasks.jsonl"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# parse_info tests
# ---------------------------------------------------------------------------


def test_parse_info_valid(info_file):
    """parse_info should return a dict with all keys from the file."""
    info = parse_info(info_file)
    assert info["codebase_version"] == "v3.0"
    assert info["fps"] == 30
    assert info["total_episodes"] == 50
    assert "features" in info
    assert info["chunks_size"] == 1000


def test_parse_info_missing_required_key(tmp_path):
    """parse_info should raise ValueError when a required key is absent."""
    for missing_key in ("codebase_version", "fps", "total_episodes", "features"):
        bad_info = {k: v for k, v in MINIMAL_INFO.items() if k != missing_key}
        p = tmp_path / f"info_no_{missing_key}.json"
        p.write_text(json.dumps(bad_info), encoding="utf-8")

        with pytest.raises(ValueError, match=missing_key):
            parse_info(p)


# ---------------------------------------------------------------------------
# parse_tasks tests
# ---------------------------------------------------------------------------


def test_parse_tasks_valid(tasks_file):
    """parse_tasks should return a list of two well-formed dicts."""
    tasks = parse_tasks(tasks_file)
    assert len(tasks) == 2
    assert tasks[0] == {"task_index": 0, "task": "Pick up the cube"}
    assert tasks[1] == {"task_index": 1, "task": "Place the cube in the bin"}


def test_parse_tasks_empty_file(tmp_path):
    """parse_tasks on an empty file should return an empty list."""
    p = tmp_path / "tasks_empty.jsonl"
    p.write_text("", encoding="utf-8")
    tasks = parse_tasks(p)
    assert tasks == []


def test_parse_tasks_whitespace_only_file(tmp_path):
    """parse_tasks on a whitespace-only file should return an empty list."""
    p = tmp_path / "tasks_ws.jsonl"
    p.write_text("\n\n\n", encoding="utf-8")
    tasks = parse_tasks(p)
    assert tasks == []


# ---------------------------------------------------------------------------
# get_video_path tests
# ---------------------------------------------------------------------------


def test_get_video_path_episode_0(info_file):
    """Episode 0 with chunks_size=1000 should land in chunk-000."""
    info = parse_info(info_file)
    result = get_video_path(info, episode_index=0, video_key="cam_high")
    assert result == "videos/chunk-000/cam_high/episode_000000.mp4"


def test_get_video_path_episode_1000(info_file):
    """Episode 1000 should land in chunk-001."""
    info = parse_info(info_file)
    result = get_video_path(info, episode_index=1000, video_key="cam_low")
    assert result == "videos/chunk-001/cam_low/episode_001000.mp4"


def test_get_video_path_episode_999(info_file):
    """Episode 999 should still be in chunk-000 (boundary check)."""
    info = parse_info(info_file)
    result = get_video_path(info, episode_index=999, video_key="cam_high")
    assert result == "videos/chunk-000/cam_high/episode_000999.mp4"


# ---------------------------------------------------------------------------
# get_data_path tests
# ---------------------------------------------------------------------------


def test_get_data_path_episode_0(info_file):
    """Episode 0 with chunks_size=1000 should land in chunk-000."""
    info = parse_info(info_file)
    result = get_data_path(info, episode_index=0)
    assert result == "data/chunk-000/episode_000000.parquet"


def test_get_data_path_episode_2500(info_file):
    """Episode 2500 should land in chunk-002."""
    info = parse_info(info_file)
    result = get_data_path(info, episode_index=2500)
    assert result == "data/chunk-002/episode_002500.parquet"


# ---------------------------------------------------------------------------
# get_camera_names tests
# ---------------------------------------------------------------------------


def test_get_camera_names(info_file):
    """get_camera_names should return only the camera suffix names, sorted."""
    info = parse_info(info_file)
    names = get_camera_names(info)
    assert names == ["cam_high", "cam_low"]


def test_get_camera_names_no_cameras(tmp_path):
    """get_camera_names should return an empty list when no camera features exist."""
    info = dict(MINIMAL_INFO)
    info["features"] = {
        "action": {"dtype": "float32", "shape": [6]},
        "timestamp": {"dtype": "float32", "shape": [1]},
    }
    assert get_camera_names(info) == []


def test_get_camera_names_single_camera(tmp_path):
    """get_camera_names should work correctly with a single camera feature."""
    info = dict(MINIMAL_INFO)
    info["features"] = {
        "observation.images.wrist_cam": {"dtype": "video", "shape": [480, 640, 3]},
        "action": {"dtype": "float32", "shape": [6]},
    }
    assert get_camera_names(info) == ["wrist_cam"]
