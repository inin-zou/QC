---
name: robotq:augment
description: Augment a LeRobot v3 robotics dataset with composable transforms (mirror, color jitter, noise, speed warp, background replace). Use when the user wants to augment, expand, multiply, or add variation to a robot dataset.
---

# Augment a LeRobot Dataset

Run a composable augmentation pipeline on a LeRobot v3 dataset and upload the result to HuggingFace Hub.

## Available Augmentations

| Name | Flag | What it does |
|------|------|-------------|
| Mirror | `--mirror` | Flip video + swap L/R arm actions (requires adapter) |
| ColorJitter | `--color-jitter` | Brightness/contrast/saturation/hue (temporally consistent) |
| GaussianNoise | `--gaussian-noise` | Per-frame pixel noise |
| ActionNoise | `--action-noise` | Gaussian perturbation on action trajectories |
| SpeedWarp | `--speed-warp` | Time-stretch episodes (0.8-1.2x) |
| BackgroundReplace | `--background "prompt"` | SD Inpainting background replacement (needs generative deps) |

## Quick Use (CLI)

```bash
robotq augment \
  --dataset lerobot/aloha_static_cups_open \
  --output USERNAME/augmented-dataset \
  --mirror --color-jitter \
  --adapter aloha \
  --multiply 2
```

## With Config File

```bash
robotq augment --config pipeline.yaml
```

```yaml
dataset: lerobot/aloha_static_cups_open
adapter: aloha
output: USERNAME/augmented-dataset
multiply: 2

pipeline:
  - type: Mirror
    p: 0.5
  - type: ColorJitter
    brightness: 0.3
    contrast: 0.2
  - type: SpeedWarp
    min_rate: 0.9
    max_rate: 1.1
```

## Python API

```python
from robotq.core.pipeline import Compose
from robotq.core.augmentations.mirror import Mirror
from robotq.core.augmentations.color import ColorJitter
from robotq.adapters.aloha import AlohaAdapter
from robotq.io.loader import load_dataset
from robotq.io.writer import write_dataset

pipeline = Compose([
    Mirror(adapter=AlohaAdapter(), p=0.5),
    ColorJitter(brightness=0.2),
])

episodes = load_dataset("lerobot/aloha_static_cups_open", max_episodes=5)
augmented = [pipeline(ep) for ep in episodes]
write_dataset(episodes + augmented, repo_id="USERNAME/augmented")
```

## Key Flags

- `--dry-run` — preview what would happen without processing
- `--no-upload` — write locally, skip Hub push
- `--preview-first` — save before/after PNGs for episode 0 before full run
- `--adapter aloha` — required for Mirror (encodes robot arm layout)
- `--multiply N` — number of augmented copies per original episode
