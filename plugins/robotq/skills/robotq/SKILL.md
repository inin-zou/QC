---
name: robotq
description: Augment LeRobot v3 robotics datasets with composable transforms (mirror, color jitter, noise, speed warp). Use when the user wants to augment, expand, or modify a LeRobot dataset.
---

# RobotQ — Dataset Augmentation Skill

Use this skill when the user asks to augment, expand, transform, or modify a LeRobot v3 dataset. This includes requests like:
- "Augment this robot dataset"
- "Double the size of my training data"
- "Mirror the aloha dataset"
- "Add noise/jitter to my dataset"
- "Create variations of this dataset"

## Available Augmentations

| Name | What it does | When to use |
|------|-------------|-------------|
| `Mirror` | Flip video horizontally + swap left/right arm actions | Bimanual robots (ALOHA). Doubles data with physically correct mirrored trajectories. |
| `ColorJitter` | Randomize brightness, contrast, saturation, hue | Always useful. Adds visual diversity. Temporally consistent (no flickering). |
| `GaussianNoise` | Add per-pixel noise to frames | Robustness to sensor noise. |
| `ActionNoise` | Add Gaussian noise to action trajectories | Robustness to control noise. |
| `SpeedWarp` | Resample episodes at different speeds | Temporal diversity. Interpolates actions/states. |

## How to Use

### Option 1: CLI (recommended for quick use)

```bash
# Simple flags
robotq augment \
  --dataset lerobot/aloha_static_cups_open \
  --output USERNAME/augmented-dataset \
  --mirror --color-jitter \
  --adapter aloha \
  --multiply 2

# With YAML config for complex pipelines
robotq augment --config pipeline.yaml
```

### Option 2: Python API (for scripting)

```python
from robotq.core.pipeline import Compose
from robotq.core.augmentations.color import ColorJitter
from robotq.core.augmentations.mirror import Mirror
from robotq.core.augmentations.noise import GaussianNoise, ActionNoise
from robotq.core.augmentations.speed import SpeedWarp
from robotq.adapters.aloha import AlohaAdapter
from robotq.io.loader import load_dataset
from robotq.io.writer import write_dataset

pipeline = Compose([
    Mirror(adapter=AlohaAdapter(), p=0.5),
    ColorJitter(brightness=0.3, contrast=0.2),
    SpeedWarp(min_rate=0.9, max_rate=1.1),
])

episodes = load_dataset("lerobot/aloha_static_cups_open", max_episodes=5)
augmented = [pipeline(ep) for ep in episodes]
write_dataset(episodes + augmented, repo_id="USERNAME/augmented-dataset")
```

### Option 3: MCP Tools (for AI agents)

If the RobotQ MCP server is configured, call the `augment_dataset` tool:

```json
{
  "source": "lerobot/aloha_static_cups_open",
  "output": "USERNAME/augmented-dataset",
  "augmentations": ["mirror", "color_jitter"],
  "adapter": "aloha",
  "multiply": 2
}
```

## YAML Config Format

For complex pipelines with probability control and composition:

```yaml
dataset: lerobot/aloha_static_cups_open
adapter: aloha
output: USERNAME/augmented-dataset
multiply: 2

pipeline:
  - type: Mirror
    p: 0.5
  - type: OneOf
    p: 0.8
    transforms:
      - type: ColorJitter
        brightness: 0.3
        contrast: 0.2
      - type: GaussianNoise
        sigma: 0.02
  - type: SpeedWarp
    min_rate: 0.8
    max_rate: 1.2
```

## Key Concepts

- **Adapter**: Robot-specific knowledge (which joints are left/right arm). Required for `Mirror`. Use `aloha` for ALOHA bimanual robots.
- **Multiply**: How many augmented copies per original episode. `--multiply 2` with 50 episodes = 150 total (50 original + 100 augmented).
- **Temporal consistency**: `ColorJitter` samples parameters once per episode (not per frame), preventing video flickering.
- **`--dry-run`**: Preview what would happen without processing.
- **`--no-upload`**: Write locally without pushing to HuggingFace Hub.

## Useful Commands

```bash
robotq list                    # Show available augmentations
robotq augment --help          # Full CLI help
robotq augment ... --dry-run   # Preview without processing
```

## Installation

```bash
git clone https://github.com/REPO/robotq
cd robotq
uv venv --python 3.12
uv pip install -e ".[dev]"

# For MCP server
uv pip install -e ".[mcp]"
```

## Important Notes

- Requires Python 3.12+ and a HuggingFace account (for uploading)
- Login first: `huggingface-cli login`
- Loading episodes requires downloading video files (can be slow on first run)
- Mirror augmentation is only meaningful for bimanual robots with an adapter
- Output is always a valid LeRobot v3 dataset compatible with the HF visualizer
