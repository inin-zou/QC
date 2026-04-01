"""Integration tests for the RobotQ dataset loader.

These tests create a real local LeRobot dataset using LeRobotDataset.create(),
populate it with synthetic frames, and then load it back through our loader.
No network access is required.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Constants shared across tests
# ---------------------------------------------------------------------------
NUM_EPISODES = 3
NUM_FRAMES = 5
IMG_H = 64
IMG_W = 64
ACTION_DIM = 4
STATE_DIM = 4
FPS = 10
CAM_NAME = "front"
ROBOT_TYPE = "test_robot"
TASK_DESC = "test task"
REPO_ID = "test/loader_dataset"


# ---------------------------------------------------------------------------
# Fixture: create a small valid local LeRobot dataset
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def local_dataset_dir(tmp_path_factory):
    """Create a minimal local LeRobot dataset for all loader tests.

    Layout:
        3 episodes × 5 frames × 1 camera (64×64 RGB video) × 4-dim action/state
        fps=10, robot_type='test_robot'

    Returns the Path to the dataset root so tests can pass it as local_dir.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset_root = tmp_path_factory.mktemp("lerobot_ds") / REPO_ID.replace("/", "_")

    features = {
        f"observation.images.{CAM_NAME}": {
            "dtype": "video",
            "shape": (IMG_H, IMG_W, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": ["action"],
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        features=features,
        root=dataset_root,
        robot_type=ROBOT_TYPE,
        use_videos=True,
    )

    rng = np.random.default_rng(seed=0)
    for _ in range(NUM_EPISODES):
        for _ in range(NUM_FRAMES):
            dataset.add_frame(
                {
                    f"observation.images.{CAM_NAME}": rng.integers(
                        0, 256, (IMG_H, IMG_W, 3), dtype=np.uint8
                    ),
                    "observation.state": rng.random(STATE_DIM).astype(np.float32),
                    "action": rng.random(ACTION_DIM).astype(np.float32),
                    "task": TASK_DESC,
                }
            )
        dataset.save_episode()

    dataset.finalize()
    return dataset_root


# ---------------------------------------------------------------------------
# Helper: load all episodes once per test session
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def loaded_episodes(local_dataset_dir):
    """Load all episodes from the local dataset via our loader."""
    from robotq.io.loader import load_dataset

    return load_dataset(REPO_ID, local_dir=str(local_dataset_dir))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_load_returns_correct_episode_count(loaded_episodes):
    assert len(loaded_episodes) == NUM_EPISODES


def test_load_max_episodes(local_dataset_dir):
    from robotq.io.loader import load_dataset

    episodes = load_dataset(REPO_ID, max_episodes=2, local_dir=str(local_dataset_dir))
    assert len(episodes) == 2


def test_episode_has_correct_frame_count(loaded_episodes):
    for ep in loaded_episodes:
        assert ep.num_frames == NUM_FRAMES


def test_episode_has_correct_action_dim(loaded_episodes):
    for ep in loaded_episodes:
        assert ep.action_dim == ACTION_DIM


def test_episode_has_correct_state_dim(loaded_episodes):
    for ep in loaded_episodes:
        assert ep.state_dim == STATE_DIM


def test_episode_metadata_fps(loaded_episodes):
    for ep in loaded_episodes:
        assert ep.metadata.fps == float(FPS)


def test_episode_metadata_robot_type(loaded_episodes):
    for ep in loaded_episodes:
        assert ep.metadata.robot_type == ROBOT_TYPE


def test_episode_metadata_camera_names(loaded_episodes):
    for ep in loaded_episodes:
        assert CAM_NAME in ep.metadata.camera_names


def test_episode_frames_are_uint8(loaded_episodes):
    for ep in loaded_episodes:
        for cam, frame_list in ep.frames.items():
            for frame in frame_list:
                assert frame.dtype == np.uint8, (
                    f"Camera '{cam}' frame dtype is {frame.dtype}, expected uint8"
                )


def test_episode_actions_are_float32(loaded_episodes):
    for ep in loaded_episodes:
        assert ep.actions.dtype == np.float32, (
            f"actions dtype is {ep.actions.dtype}, expected float32"
        )
