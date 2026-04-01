---
name: robotq:inspect
description: Inspect a LeRobot v3 dataset — show episode count, frame dimensions, action/state shape, camera names, FPS, and task descriptions.
---

# Inspect a LeRobot Dataset

Load a dataset and display its structure without augmenting.

## CLI

```bash
robotq augment \
  --dataset lerobot/aloha_static_cups_open \
  --output dummy \
  --max-episodes 1 \
  --dry-run
```

The dry-run output shows: episode count, frames per episode, action/state dimensions, camera names, robot type, FPS.

## Python API (more detailed)

```python
from robotq.io.loader import load_dataset

episodes = load_dataset("lerobot/aloha_static_cups_open", max_episodes=1)
ep = episodes[0]

print(f"Frames: {ep.num_frames}")
print(f"Cameras: {ep.metadata.camera_names}")
print(f"Action dim: {ep.action_dim}")
print(f"State dim: {ep.state_dim}")
print(f"FPS: {ep.metadata.fps}")
print(f"Robot: {ep.metadata.robot_type}")
print(f"Task: {ep.metadata.task_description}")
print(f"Frame shape: {ep.frames[ep.metadata.camera_names[0]][0].shape}")
```

## Other Useful Commands

```bash
robotq list       # Show available augmentations
robotq adapters   # Show available robot adapters
```
