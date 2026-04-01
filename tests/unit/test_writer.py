"""Unit tests for robotq.io.writer."""

import numpy as np
import pytest

from robotq.core.episode import Episode, EpisodeMetadata
from robotq.io.writer import generate_visualizer_link, write_dataset


# ---------------------------------------------------------------------------
# generate_visualizer_link tests
# ---------------------------------------------------------------------------


def test_generate_visualizer_link_default_episode():
    """Default episode=0 should produce the correct encoded URL."""
    url = generate_visualizer_link("user/dataset")
    expected = (
        "https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2Fuser%2Fdataset%2Fepisode_0"
    )
    assert url == expected


def test_generate_visualizer_link_episode_5():
    """episode=5 should appear in the URL."""
    url = generate_visualizer_link("user/dataset", episode=5)
    expected = (
        "https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2Fuser%2Fdataset%2Fepisode_5"
    )
    assert url == expected


def test_generate_visualizer_link_complex_repo_id():
    """A repo_id with multiple slashes encodes each one."""
    url = generate_visualizer_link("org/sub/dataset", episode=0)
    assert "%2Forg%2Fsub%2Fdataset%2F" in url


# ---------------------------------------------------------------------------
# write_dataset tests
# ---------------------------------------------------------------------------


def _make_episode(
    episode_index: int,
    num_frames: int = 5,
    height: int = 64,
    width: int = 64,
    action_dim: int = 4,
    state_dim: int = 4,
    camera_names: list[str] | None = None,
    task: str = "pick up the cube",
    fps: float = 30.0,
) -> Episode:
    """Create a minimal synthetic Episode for testing."""
    if camera_names is None:
        camera_names = ["cam_top"]

    frames = {
        cam: [
            np.random.randint(0, 255, (height, width, 3), dtype=np.uint8) for _ in range(num_frames)
        ]
        for cam in camera_names
    }
    actions = np.random.randn(num_frames, action_dim).astype(np.float32)
    states = np.random.randn(num_frames, state_dim).astype(np.float32)
    metadata = EpisodeMetadata(
        episode_index=episode_index,
        task_description=task,
        task_id=0,
        fps=fps,
        camera_names=camera_names,
        robot_type="test_robot",
    )
    return Episode(frames=frames, actions=actions, states=states, metadata=metadata)


def test_write_dataset_can_be_imported():
    """Smoke test: write_dataset is importable and callable."""
    assert callable(write_dataset)


def test_write_dataset_rejects_empty_episodes():
    """write_dataset should raise ValueError for an empty episode list."""
    with pytest.raises(ValueError, match="episodes list must not be empty"):
        write_dataset([], repo_id="test/empty", local_only=True)


def test_write_dataset_local_only(tmp_path):
    """write_dataset with local_only=True creates a valid dataset on disk.

    This is the primary integration-style test.  It creates 2 synthetic
    episodes (5 frames each, 1 camera at 64x64, 4-dim actions/states)
    and verifies that the dataset was written without error.
    """
    repo_id = "test_user/test_dataset"
    root = tmp_path / repo_id

    episodes = [_make_episode(i) for i in range(2)]
    url = write_dataset(
        episodes,
        repo_id=repo_id,
        root=root,
        local_only=True,
    )

    # The return value is a visualizer link
    assert "huggingface.co" in url
    assert "test_user" in url
    assert "test_dataset" in url

    # The dataset directory should exist and contain info.json
    assert root.exists()
    assert (root / "meta" / "info.json").exists()


def test_write_dataset_local_only_two_cameras(tmp_path):
    """write_dataset handles multi-camera episodes."""
    repo_id = "test_user/multicam"
    root = tmp_path / repo_id

    episodes = [_make_episode(i, camera_names=["cam_high", "cam_low"]) for i in range(2)]
    url = write_dataset(
        episodes,
        repo_id=repo_id,
        root=root,
        local_only=True,
    )

    assert root.exists()
    assert (root / "meta" / "info.json").exists()

    # Verify the link is well-formed
    assert "huggingface.co" in url
