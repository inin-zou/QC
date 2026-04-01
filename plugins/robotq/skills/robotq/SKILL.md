---
name: robotq
description: Overview of RobotQ — a composable augmentation toolkit for LeRobot v3 robotics datasets. Use this to understand what RobotQ can do and which sub-skill to invoke.
---

# RobotQ — Dataset Augmentation Toolkit

Composable augmentation toolkit for LeRobot v3 datasets with action-aware transforms.

## Sub-Skills

Use these specific skills for targeted tasks:

| Skill | Command | When to use |
|-------|---------|-------------|
| `robotq:augment` | `/robotq:augment` | Augment a dataset (CLI, Python API, or config file) |
| `robotq:preview` | `/robotq:preview` | Preview augmentation on one episode before full run |
| `robotq:configure` | `/robotq:configure` | Generate a YAML pipeline config |
| `robotq:inspect` | `/robotq:inspect` | Inspect a dataset's structure (episodes, cameras, actions) |

## Quick Reference

```bash
# Augment with mirror + color jitter
robotq augment --dataset lerobot/aloha_static_cups_open \
  --output USERNAME/augmented --mirror --color-jitter --adapter aloha --multiply 2

# Preview first
robotq preview --dataset lerobot/aloha_static_cups_open --mirror --adapter aloha

# List augmentations
robotq list

# List adapters
robotq adapters
```

## Available Augmentations

| Name | Type | Description |
|------|------|-------------|
| Mirror | RobotTransform | Flip video + swap L/R arm actions via adapter |
| ColorJitter | SequenceTransform | Brightness/contrast/saturation/hue (temporally consistent) |
| GaussianNoise | FrameTransform | Per-frame pixel noise |
| ActionNoise | TrajectoryTransform | Gaussian perturbation on actions |
| SpeedWarp | TrajectoryTransform | Time-stretch episodes |
| BackgroundReplace | TrajectoryTransform | SD Inpainting background replacement |

## Available Adapters

| Name | Robot Type | Mirror Behavior |
|------|-----------|-----------------|
| `aloha` | ALOHA bimanual (14-DOF) | Swaps L/R arm actions + negates waist joints |
| `generic` | Any single-arm | Flips video only, actions/states unchanged |
