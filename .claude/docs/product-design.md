# RobotQ — Product Design

## Identity

**One-liner:** A composable augmentation toolkit for LeRobot v3 datasets — CLI-first, MCP-ready, with generative superpowers.

**Name:** RobotQ (the "Q" ties to Qualia)

**Narrative:** "Python for ecosystem integration and shipping speed, Rust for performance-critical augmentation kernels."

## Problem

LeRobot v3 datasets have no tool for **persistent, paired augmentation** that outputs a new dataset. The gaps:

- LeRobot's built-in transforms are training-time only — ephemeral, never saved
- `lerobot-edit-dataset` handles split/merge/delete, not augmentation
- AugLab (hackathon project) does image-only augmentation with no LeRobot v3 format integration, no action awareness, no CLI, no Hub upload
- Albumentations is image-only — no action/trajectory/robotics awareness
- No existing MCP tool for dataset augmentation

**The critical gap:** No existing tool does paired image+action augmentation as a composable pipeline that outputs a new LeRobot v3 dataset.

## Target User

Robotics ML engineers who want to expand their training datasets without collecting more real-world demonstrations.

## What It Does

1. Takes any LeRobot v3 dataset from HF Hub (or local)
2. Applies a composable pipeline of augmentations (visual + action-aware + generative)
3. Outputs a new LeRobot v3 dataset, uploaded to HF Hub
4. Prints a visualizer link

## Distribution Layers (built in order)

1. **Python library** — `from robotq import Pipeline, Mirror, BackgroundReplace`
2. **CLI** — `robotq augment --mirror --background "kitchen" lerobot/aloha_static_cups_open`
3. **MCP server** (Phase 2) — AI agents can call `augment_dataset` as a tool
4. **Claude Code skill** (Phase 2) — opinionated defaults for agent-driven augmentation

## Augmentation Catalog

### Core (no external models)

| Augmentation | Type | What it touches | Description |
|---|---|---|---|
| Mirror | RobotTransform | Frames + actions + states | Flip video horizontally, swap L/R arm actions via adapter |
| ColorJitter | SequenceTransform | Frames only | Brightness, contrast, saturation, hue (consistent per episode) |
| GaussianNoise | FrameTransform | Frames only | Per-frame pixel noise |
| ActionNoise | TrajectoryTransform | Actions only | Gaussian perturbation on action trajectories |
| SpeedWarp | TrajectoryTransform | Full episode | Resample episode at different FPS with interpolation |

### Generative (optional install: `uv pip install robotq[generative]`)

| Augmentation | Type | Models | Description |
|---|---|---|---|
| BackgroundReplace | SequenceTransform | SAM2 + SD Inpainting | Segment robot+objects, inpaint new background from text prompt |

Background replacement supports two methods:
- `auto` — SAM2 segmentation (accurate, needs GPU)
- `fast` — Motion-based mask via frame differencing (rough, CPU-only)

## Adapter System

Robotics augmentation requires knowing the action schema. Different robots have different joint layouts, coordinate systems, and symmetry properties.

RobotQ uses **adapters** to encode this knowledge:

```python
mirror = Mirror(adapter=AlohaAdapter(), p=0.5)
```

Built-in adapters:
- `AlohaAdapter` — ALOHA bimanual (14-DOF, 7 left + 7 right)

The adapter pattern makes schema dependency explicit rather than pretending augmentations are universal.

## CLI Interface

### Commands

| Command | Purpose |
|---------|---------|
| `robotq augment` | Load dataset, build pipeline, run transforms, write output, optionally upload |
| `robotq preview` | Preview augmentation on a single episode, save before/after comparison |
| `robotq list` | List available augmentations with type, adapter requirements, dependencies |
| `robotq adapters` | List available robot adapters and supported transforms |

### Two Usage Modes

**Simple mode — flags for quick use:**
```bash
robotq augment \
  --dataset lerobot/aloha_static_cups_open \
  --color-jitter \
  --gaussian-noise \
  --output yourname/aloha-aug-basic
```

**Advanced mode — config file for composable pipelines:**
```bash
robotq augment --config pipeline.yaml
```

Config file expresses the full Compose/OneOf/SomeOf pipeline:
```yaml
dataset: lerobot/aloha_static_cups_open
adapter: aloha_bimanual
output: yourname/aloha_aug_v1
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

Config is better than flags because: it can express OneOf/SomeOf/probabilities, it's easier for AI agents to generate, and it's self-documenting in the repo.

### augment Parameters

**Input/output:**
- `--dataset` — HF repo ID or local path
- `--output` — Output HF repo ID
- `--local-path` — Local dataset path (alternative to --dataset)

**Pipeline definition (simple mode):**
- `--mirror`, `--color-jitter`, `--gaussian-noise`, `--speed-warp`, `--action-noise`
- `--background "prompt"` — generative background replacement
- `--background-method auto|fast`

**Pipeline definition (advanced mode):**
- `--config pipeline.yaml`

**Run control:**
- `--multiply N` — dataset multiplication factor
- `--adapter NAME` — robot adapter (e.g., aloha_bimanual)
- `--dry-run` — parse everything, estimate output, don't process or upload
- `--no-upload` — write locally but skip push_to_hub
- `--preview-first` — preview episode 0 before full run

**UX:**
- `--verbose` — detailed logging
- `--plain` — plain text output (no rich formatting, good for CI/logs)

### preview Output

```bash
robotq preview \
  --dataset lerobot/aloha_static_cups_open \
  --episode 0 \
  --config pipeline.yaml
```

Produces:
- Before/after frame PNGs saved to `preview/` directory
- Action/state diff summary printed to terminal
- Side-by-side comparison for each camera view

Minimum viable preview: a few sampled frames as before/after PNGs + file paths printed.

### list Output

```
$ robotq list

Available augmentations:

Name              Type             Adapter    Dependencies
Mirror            RobotTransform   required   built-in
ColorJitter       SequenceImage    -          built-in
GaussianNoise     FrameImage       -          built-in
ActionNoise       Trajectory       -          built-in
SpeedWarp         Trajectory       -          built-in
BackgroundReplace SequenceImage    -          optional[generative]
```

### adapters Output

```
$ robotq adapters

Available adapters:

Name              Robot Type       Mirror Support   DOF
aloha_bimanual    ALOHA            L/R arm swap     14
```

### dry-run Output

```
$ robotq augment --config pipeline.yaml --dry-run

Dataset: lerobot/aloha_static_cups_open
Episodes found: 50
Adapter: aloha_bimanual

Pipeline:
  1. Mirror(p=0.5)
  2. OneOf(ColorJitter, GaussianNoise, p=0.8)
  3. SpeedWarp(0.8-1.2)

Multiply factor: 2
Expected output episodes: 100
Upload target: yourname/aloha_aug_v1
Status: dry-run only, no files written
```

## Python API (alongside CLI)

The library is usable directly from Python — not just a CLI wrapper:

```python
from robotq.core.pipeline import Compose
from robotq.core.augmentations import ColorJitter, SpeedWarp, Mirror
from robotq.adapters.aloha import AlohaAdapter
from robotq.io.loader import load_dataset
from robotq.io.writer import write_dataset

pipeline = Compose([
    Mirror(adapter=AlohaAdapter(), p=0.5),
    ColorJitter(brightness=0.2),
    SpeedWarp(min_rate=0.9, max_rate=1.1),
])

episodes = load_dataset("lerobot/aloha_static_cups_open")
augmented = [pipeline(ep) for ep in episodes]
write_dataset(augmented, repo_id="yourname/aloha_aug_v1")
```

This makes RobotQ a reusable library, not just a script.

## Demo Story

**Input:** `lerobot/aloha_static_cups_open` — 50 episodes of a bimanual robot opening cups, 4 camera views.

**Pipeline:** Mirror (doubles to 100 episodes) + ColorJitter (lighting variation) + optional BackgroundReplace ("industrial kitchen")

**Output:** 100-300 augmented episodes uploaded to HF Hub with direct visualizer link.

**Hero screenshot:** Same robot, same task, three different visual environments side-by-side. Before/after that makes someone stop scrolling.

**Demo terminal output:**
```
$ robotq augment --config examples/aloha_basic.yaml

Loading dataset: lerobot/aloha_static_cups_open
Found 50 episodes
Adapter: aloha_bimanual

Building augmentation pipeline
  - Mirror(p=0.5)
  - OneOf(ColorJitter, GaussianNoise, p=0.8)
  - SpeedWarp(min_rate=0.8, max_rate=1.2)

Processing episodes
  [################################] 50/50 episodes

Writing augmented dataset
  Output repo: yourname/aloha_static_cups_open_aug
  Episodes written: 100

Uploading to Hugging Face Hub
  Upload complete

Visualizer:
  https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2Fyourname%2Faloha_static_cups_open_aug%2Fepisode_0
```

## Tech Stack

- **Python 3.12+** — main language, matches LeRobot's requires-python
- **uv** — dependency management
- **Polars** — parquet reading (Rust-backed, fast)
- **OpenCV** — video decoding only (LeRobot handles encoding)
- **Typer + Rich** — CLI framework
- **Ruff** — lint + format
- **LeRobot** — dataset format, LeRobotDataset API for read/write (official writer guarantees valid output)
- **SAM2 + Stable Diffusion Inpainting** — generative augmentation (optional dep)
- **PyO3 + Maturin** — Rust extension for frame processing kernels (Phase 2, stretch)

## Competitive Positioning

| Feature | AugLab | Albumentations | lerobot-edit-dataset | **RobotQ** |
|---|---|---|---|---|
| LeRobot v3 native | No | No | Yes | **Yes** |
| Action-aware augmentation | No | No | No | **Yes** |
| Composable pipeline | No | Yes | No | **Yes** |
| Persistent output (new dataset) | Export only | No | Yes (edit) | **Yes** |
| HF Hub upload | No | No | No | **Yes** |
| Generative augmentation | No | No | No | **Yes** |
| CLI | No | No | Yes | **Yes** |
| MCP server | No | No | No | **Phase 2** |

## Build Priority

### MVP (must ship)
- Dataset loader + writer + video codec
- Episode container + pipeline + transform base classes
- ColorJitter, GaussianNoise, ActionNoise
- Mirror + AlohaAdapter
- CLI `augment` command (flags + `--config pipeline.yaml`)
- `--dry-run` support
- `preview` command (before/after PNGs)
- `list` command (with type/adapter/dep info)
- push_to_hub + visualizer link
- Python API examples in README
- Rich output by default

### Stretch Goals (in priority order)
- SpeedWarp
- `adapters` command
- `--plain` output mode
- BackgroundReplace (generative)
- Rust kernel for frame processing
- MCP server
- Claude Code skill
