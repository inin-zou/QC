# RobotQ

Composable augmentation toolkit for LeRobot v3 datasets. Augments robotics training data with action-aware transforms and uploads to HuggingFace Hub.

**Live demo dataset:** [YongkangZOU/aloha-robotq-demo](https://huggingface.co/datasets/YongkangZOU/aloha-robotq-demo) — [Visualize](https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2FYongkangZOU%2Faloha-robotq-demo%2Fepisode_0)

## What It Does

RobotQ takes a LeRobot v3 dataset, applies a composable pipeline of augmentations (visual, action-aware, and temporal), and outputs a new valid LeRobot v3 dataset on HuggingFace Hub.

**Key differentiator:** Unlike image-only augmentation tools, RobotQ understands that flipping a robot video horizontally requires also swapping left/right arm actions and negating mirror-sensitive joint axes. This is done through an adapter system that encodes robot-specific schema knowledge.

### Available Augmentations

| Name | Type | What It Does |
|------|------|-------------|
| Mirror | RobotTransform | Flip video + swap L/R arm actions via adapter |
| ColorJitter | SequenceTransform | Brightness/contrast/saturation/hue (consistent per episode) |
| GaussianNoise | FrameTransform | Per-frame pixel noise |
| ActionNoise | TrajectoryTransform | Gaussian perturbation on action trajectories |
| SpeedWarp | TrajectoryTransform | Time-stretch episodes with interpolated actions |

### Three Interfaces

- **CLI** — `robotq augment`, `robotq list`, `robotq preview`
- **Python API** — `from robotq.core.pipeline import Compose`
- **MCP Server** — AI agents can call `augment_dataset` as a tool

## How to Run

### Installation

```bash
git clone https://github.com/YongkangZOU/QC.git
cd QC
uv venv --python 3.12
uv pip install -e ".[dev]"
```

Login to HuggingFace (for uploading):
```bash
huggingface-cli login
```

### Quick Start (CLI)

```bash
# Augment with mirror + color jitter, upload to Hub
robotq augment \
  --dataset lerobot/aloha_static_cups_open \
  --output YOUR_USERNAME/aloha-augmented \
  --mirror --color-jitter \
  --adapter aloha \
  --multiply 2

# Preview before committing
robotq augment ... --dry-run

# Save before/after PNGs
robotq preview \
  --dataset lerobot/aloha_static_cups_open \
  --mirror --color-jitter --adapter aloha

# List available augmentations
robotq list
```

### Config File (for complex pipelines)

```bash
robotq augment --config examples/aloha_basic.yaml
```

```yaml
# examples/aloha_basic.yaml
dataset: lerobot/aloha_static_cups_open
adapter: aloha
output: YOUR_USERNAME/aloha-augmented
multiply: 2

pipeline:
  - type: Mirror
    p: 0.5
  - type: ColorJitter
    brightness: 0.3
    contrast: 0.2
    p: 1.0
```

### Python API

```python
from robotq.core.pipeline import Compose
from robotq.core.augmentations.color import ColorJitter
from robotq.core.augmentations.mirror import Mirror
from robotq.core.augmentations.speed import SpeedWarp
from robotq.adapters.aloha import AlohaAdapter
from robotq.io.loader import load_dataset
from robotq.io.writer import write_dataset

pipeline = Compose([
    Mirror(adapter=AlohaAdapter(), p=0.5),
    ColorJitter(brightness=0.2),
    SpeedWarp(min_rate=0.9, max_rate=1.1),
])

episodes = load_dataset("lerobot/aloha_static_cups_open", max_episodes=5)
augmented = [pipeline(ep) for ep in episodes]
write_dataset(episodes + augmented, repo_id="YOUR_USERNAME/aloha-augmented")
```

### MCP Server (for AI agents)

```bash
uv pip install -e ".[mcp]"
# Add to your .mcp.json or Claude Code config
```

AI agents can then call `augment_dataset`, `list_augmentations`, and `preview_augmentation` as tools.

## Architecture

```
robotq/
├── core/                       # Engine layer
│   ├── episode.py              # Episode dataclass (universal data container)
│   ├── transform.py            # 4 base classes: Frame, Sequence, Trajectory, Robot
│   ├── pipeline.py             # Compose, OneOf, SomeOf (Albumentations-style)
│   ├── config.py               # YAML config -> pipeline builder
│   └── augmentations/          # Mirror, ColorJitter, Noise, SpeedWarp
├── io/                         # I/O layer
│   ├── loader.py               # LeRobot API + OpenCV video decoding
│   ├── writer.py               # LeRobot official writer (guarantees valid output)
│   ├── video.py                # Pure MP4 decoder (decode only)
│   └── schema.py               # JSON/JSONL metadata parsing
├── adapters/                   # Robot-specific knowledge
│   ├── base.py                 # ActionAdapter protocol
│   └── aloha.py                # ALOHA bimanual (14-DOF, L/R arm swap)
├── cli/main.py                 # Typer CLI (augment, preview, list)
├── mcp/server.py               # MCP server for AI agent integration
└── skill/robotq.md             # Claude Code skill
```

**Design choices:**
- **Episode is the universal container** — every transform takes and returns an Episode. No loose dicts.
- **4 transform base classes** — FrameTransform (per-frame random), SequenceTransform (temporally consistent), TrajectoryTransform (episode-level), RobotTransform (paired image+action via adapter).
- **LeRobot official writer** — we intentionally rely on `LeRobotDataset.create()` to guarantee output compatibility with the HF visualizer and training tools.
- **OpenCV for video decoding** — bypasses torchcodec FFmpeg dependency issues; frame-range seeking for efficiency.

## How AI Coding Agents Were Used

This project was built almost entirely through AI agent orchestration using Claude Code. The workflow:

### Brainstorming Phase
- Collaborative product design session exploring competitive landscape (AugLab, Albumentations, RoboEngine)
- Produced 6 design documents: product-design.md, architecture.md, design-patterns.md, ROADMAP.md, agentic-engineering.md, test.md

### Implementation via Parallel Agent Dispatch
The agentic-engineering.md playbook defined exact agent dispatch blocks for each phase:

**Phase 1A** — 3 agents in parallel (worktree isolation):
- Agent 1: Episode container + tests
- Agent 2: Video decoder + tests
- Agent 3: Schema parser + tests

**Phase 1B** — 2 agents in parallel:
- Agent 4: Dataset loader + tests
- Agent 5: Dataset writer + tests

**Phase 2A** — 3 agents in parallel:
- Agent 6: Noise augmentations + tests
- Agent 7: Adapter system + tests
- Agent 8: Pipeline composition + tests

**Phase 3A** — 3 agents in parallel:
- Agent 9: CLI (Typer + Rich)
- Agent 10: Config parser + YAML examples
- Agent 11: README

**Phase 4** — 2 agents in parallel:
- Agent 12: SpeedWarp augmentation
- Agent 13: MCP server

### Quality Gates
- **Validation gates** between phases: full test suite + import smoke tests
- **Error handling review**: dedicated code-reviewer agent found 8 issues (3 critical), all fixed
- **Integration testing**: real aloha dataset loaded, augmented, and pushed to Hub at each phase boundary

### What the Agentic Workflow Caught
1. **LeRobot v3.0 format mismatch** — initial research said per-episode files, but real datasets pack all episodes into single files with chunk/file indexing. Discovered during Phase 1C integration, loader rewritten from scratch.
2. **torchcodec FFmpeg dependency** — LeRobot's default video decoder failed on macOS. Switched to OpenCV with frame-range seeking.
3. **push_to_hub API change** — method lives on `LeRobotDataset`, not `LeRobotDatasetMetadata`. Found during real Hub upload test.
4. **Silent frame padding** — code-reviewer agent flagged the loader silently padding frames when video/parquet frame counts disagreed. Changed to log warning + raise on zero frames.

### Stats
- **15+ agent dispatches** across 4 phases
- **117 unit tests** written by agents + manual integration
- **~3,200 lines of code** (excluding tests)
- Design docs written before any code — agents followed the spec

## Testing

```bash
# Run all unit tests
uv run pytest tests/unit/ -v

# Quick smoke test
robotq list
robotq augment --dataset lerobot/aloha_static_cups_open --output test/smoke \
  --color-jitter --adapter aloha --max-episodes 1 --no-upload --dry-run
```

117 tests covering: Episode validation, video decoding, schema parsing, all 5 augmentations, pipeline composition, adapter arm-swap logic, config parsing, writer integrity checks.

## Roadmap

- BackgroundReplace — SAM2 + Stable Diffusion Inpainting for generative scene changes
- Rust kernels — PyO3 acceleration for frame processing hot loops
- More robot adapters — single-arm, mobile manipulators
- Training integration — direct use as a LeRobot training-time transform
