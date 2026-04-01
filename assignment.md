# RobotQ

Composable augmentation toolkit for LeRobot v3 datasets — CLI-first, with action-aware transforms.

---

## Key Features

- **Composable pipeline** — `Compose`, `OneOf`, `SomeOf` operators inspired by Albumentations, applicable to full robotics episodes
- **Action-aware augmentation** — `Mirror` swaps left/right arm actions in the correct joint dimensions, not just pixels
- **Adapter system** — robot-specific schema knowledge is encoded in adapters (e.g., `AlohaAdapter`) and injected at pipeline build time, making schema dependency explicit and extensible
- **LeRobot v3 native** — reads and writes valid LeRobot v3 datasets via the official `LeRobotDataset` API
- **HuggingFace Hub integration** — auto-uploads the output dataset and prints a direct visualizer link
- **Temporal consistency** — `SequenceTransform` subclasses sample parameters once per episode so lighting and color remain stable across frames, preventing flickering

---

## Installation

```bash
git clone https://github.com/yourname/robotq
cd robotq
uv venv --python 3.12
uv pip install -e ".[dev]"
```

For generative augmentations (BackgroundReplace — requires GPU):

```bash
uv pip install -e ".[generative]"
```

---

## Quick Start (CLI)

Simple flag mode — one flag per augmentation:

```bash
robotq augment \
  --dataset lerobot/aloha_static_cups_open \
  --output yourname/aloha-augmented \
  --mirror --color-jitter \
  --adapter aloha --multiply 2
```

This loads the source dataset, applies Mirror and ColorJitter, writes 2x the original episode count, uploads to HuggingFace Hub, and prints a visualizer link.

---

## Advanced Usage (config file)

For composable pipelines with `OneOf`/`SomeOf` and per-transform probabilities, use a YAML config:

```bash
robotq augment --config pipeline.yaml
```

`pipeline.yaml`:

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

Config files can express arbitrary nesting, probabilities, and multiple augmentation branches. They are also easier for AI agents to generate and self-documenting in the repo.

Other useful commands:

```bash
# Preview augmentation on episode 0 (saves before/after PNGs to preview/)
robotq preview --dataset lerobot/aloha_static_cups_open --episode 0 --config pipeline.yaml

# List available augmentations
robotq list

# List available robot adapters
robotq adapters

# Dry run: validate config and estimate output without processing
robotq augment --config pipeline.yaml --dry-run
```

---

## Python API

RobotQ is a library first. The CLI is a thin layer over the same objects you can import directly:

```python
from robotq.core.pipeline import Compose
from robotq.core.augmentations.color import ColorJitter
from robotq.core.augmentations.mirror import Mirror
from robotq.adapters.aloha import AlohaAdapter
from robotq.io.loader import load_dataset
from robotq.io.writer import write_dataset

pipeline = Compose([
    Mirror(adapter=AlohaAdapter(), p=0.5),
    ColorJitter(brightness=0.2),
])

episodes = load_dataset("lerobot/aloha_static_cups_open", max_episodes=5)
augmented = [pipeline(ep) for ep in episodes]
write_dataset(episodes + augmented, repo_id="yourname/aloha-aug")
```

---

## Available Augmentations

| Name | Type | What it touches | Adapter required | Description |
|------|------|-----------------|------------------|-------------|
| Mirror | RobotTransform | Frames + actions + states | Yes | Horizontal flip of all camera frames, L/R arm joint swap and axis sign correction via adapter |
| ColorJitter | SequenceTransform | Frames | No | Brightness, contrast, saturation, hue — params sampled once per episode for temporal consistency |
| GaussianNoise | FrameTransform | Frames | No | Per-frame additive Gaussian pixel noise — each frame gets independent noise |
| ActionNoise | TrajectoryTransform | Actions | No | Gaussian perturbation applied to the full action trajectory |
| SpeedWarp | TrajectoryTransform | Full episode | No | Resamples episode at a different effective FPS with interpolated frames and actions |

Generative (optional install):

| Name | Type | What it touches | Adapter required | Description |
|------|------|-----------------|------------------|-------------|
| BackgroundReplace | SequenceTransform | Frames | No | Segments robot and objects via SAM2, inpaints a new background from a text prompt using Stable Diffusion |

---

## Architecture

RobotQ is structured as four layers with strict separation of concerns:

```
core/          Transform base classes, Episode container, pipeline operators, augmentations
io/            Dataset loader, writer, video decoder, schema parser
cli/           Typer app — thin translation from flags/config to core/io calls
adapters/      Robot-specific action schema knowledge
```

The `Episode` dataclass is the universal data container — every transform receives and returns an `Episode` with frames, actions, states, and metadata. No loose dicts are passed across module boundaries.

**Four transform base classes** enforce the correct dispatch pattern for each augmentation type:

- `FrameTransform` — applies independently to each frame with different random params (prevents identical per-frame artifacts)
- `SequenceTransform` — samples params once per episode, applies uniformly across all frames (prevents temporal flickering)
- `TrajectoryTransform` — operates on the full episode at once; may change episode length, FPS, or metadata
- `RobotTransform` — paired image and action transform; requires an `ActionAdapter` for robot schema awareness

Pipeline operators (`Compose`, `OneOf`, `SomeOf`) are themselves `Transform` subclasses, so they nest arbitrarily. YAML configs map 1:1 to Python objects via a registry, enabling both human and agent-authored pipelines.

The I/O layer is split into four non-overlapping modules: `loader.py` (read Hub or local), `video.py` (decode-only MP4), `schema.py` (parse info.json and tasks.jsonl), and `writer.py` (create and push LeRobotDataset). `writer.py` uses the official `LeRobotDataset.create() → add_frame() → save_episode() → finalize() → push_to_hub()` write path, guaranteeing valid v3 output.

---

## How AI Coding Agents Were Used

RobotQ was built using a structured agentic workflow with 15 total agent dispatches across 3 phases.

**Workflow overview:**

1. **Brainstorming** — a brainstorming agent explored the problem space (what gaps exist in the LeRobot ecosystem, what the MVP should be), producing a product brief.
2. **Design documents** — separate agents drafted `product-design.md`, `architecture.md`, `design-patterns.md`, and `agentic-engineering.md`. These docs became the source of truth for all subsequent agents.
3. **Implementation plan** — `agentic-engineering.md` encodes a detailed dispatch plan: which agents to run in parallel, what each writes, what tests must pass before the next phase begins.
4. **Parallel execution** — agents were dispatched in parallel batches using worktree isolation so they could work on independent modules without conflicting.

**Phase 1 (Foundation) — 5 agents + 1 review:**
- Three agents ran simultaneously: Episode container (with unit tests), video decoder, and schema parser.
- Two more agents ran in parallel once Phase 1A passed its gate: dataset loader and dataset writer.
- A code-reviewer agent then checked all Phase 1 files against `design-patterns.md`.

**Phase 2 (Core Augmentations) — 4 agents + 1 review:**
- Four agents ran simultaneously: full transform base classes, noise augmentations (GaussianNoise + ActionNoise), adapter protocol and AlohaAdapter, and pipeline composition operators.
- Mirror was implemented manually after Phase 2A, as it required careful integration of adapter + transform + video and served as a demo checkpoint.
- A code-reviewer agent checked Phase 2 output.

**Phase 3 (CLI and Polish) — 3 agents + 1 review:**
- Three agents ran simultaneously: CLI `augment`/`list` commands, YAML config parsing and pipeline registry, and README.
- A final code-reviewer agent reviewed the full codebase.

**Test-first development:** Every agent prompt included explicit instructions to write tests alongside implementation code and to run them before marking the task complete. Tests were a precondition for advancing to the next phase, not an afterthought.

**Review gates:** After each phase, a dedicated code-reviewer agent checked the output against the design patterns document — catching issues like modules violating I/O separation boundaries and missing error handling at validation boundaries.

**Agentic workflow caught real issues:** The review after Phase 1 flagged that the initial loader draft was passing loose dicts rather than Episode objects across a module boundary. The review after Phase 2 identified missing `p`-check logic in one transform's `apply()` before it reached integration testing. These were caught before they could propagate.

---

## Roadmap

- **SpeedWarp** — time-stretch augmentation with frame and action interpolation
- **BackgroundReplace** — SAM2 segmentation + Stable Diffusion inpainting for background replacement
- **MCP server** — expose `augment_dataset`, `preview_augmentation`, and `list_augmentations` as MCP tools so AI agents can augment datasets directly
- **Rust kernels** — SIMD-accelerated frame flip and color jitter via PyO3/Maturin for batch episode processing performance
- **More robot adapters** — single-arm manipulators, mobile manipulators, and custom DOF configurations
