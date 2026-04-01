# RobotQ — Roadmap

## Timeline

- **Start:** 13:00
- **Now:** ~15:00
- **Deadline:** 19:00
- **Remaining:** ~4 hours

## Principle: Demo-First

The hero moment is **clicking a visualizer link and seeing augmented episodes with correct robot actions**. Everything else is polish. Each phase ends with a working, demoable state.

---

## Phase 1: Foundation + First Demo (15:00 - 16:30, ~90 min)

**Goal:** Load a dataset, augment one episode with ColorJitter, write it back, push to Hub, print visualizer link. This is the minimum viable demo.

### 1.1 Project scaffold (15 min)
- [ ] pyproject.toml with deps (lerobot, polars, numpy, opencv-python-headless, typer, rich, huggingface_hub, pyyaml) via uv
- [ ] Package structure: robotq/core/, robotq/io/, robotq/adapters/, robotq/cli/
- [ ] Verify `uv pip install -e .` works

### 1.2 Episode container (10 min)
- [ ] `core/episode.py` — Episode + EpisodeMetadata dataclasses
- [ ] Unit test: create Episode, verify fields

### 1.3 Video Decoder (15 min) — HIGHEST RISK
- [ ] `io/video.py` — decode MP4 → list of numpy frames (decode only; LeRobot handles encoding)
- [ ] Unit test: decode a test video → verify frame count + shape + RGB order
- [ ] Test with actual aloha dataset video file

### 1.4 Dataset loader (20 min)
- [ ] `io/schema.py` — parse info.json, tasks.jsonl
- [ ] `io/loader.py` — download dataset from Hub, iterate episodes as Episode objects
- [ ] Unit test: load aloha dataset, verify episode count, frame shapes, action dims

### 1.5 Dataset writer + Hub upload (20 min)
- [ ] `io/writer.py` — create new LeRobotDataset, write episodes, finalize, push_to_hub
- [ ] Generate visualizer link
- [ ] **CHECKPOINT: round-trip test** — load dataset → write unchanged → push → click visualizer link → verify it works

### 1.6 First augmentation: ColorJitter (10 min)
- [ ] `core/transform.py` — SequenceTransform base class
- [ ] `core/augmentations/color.py` — ColorJitter implementation
- [ ] Unit test: apply to episode, verify frames changed, actions unchanged
- [ ] **DEMO 1:** Load aloha → ColorJitter → push → visualizer link works

---

## Phase 2: Core Augmentations + Mirror (16:30 - 17:30, ~60 min)

**Goal:** Ship Mirror (the robotics-meaningful augmentation) and basic pipeline composition. This is the "real" demo.

### 2.1 Transform base classes (10 min)
- [ ] `core/transform.py` — FrameTransform, SequenceTransform, TrajectoryTransform, RobotTransform
- [ ] Unit test: verify each base class dispatch pattern

### 2.2 Pipeline composition (15 min)
- [ ] `core/pipeline.py` — Compose, OneOf, SomeOf
- [ ] Unit test: Compose applies in order, OneOf picks one, probability works

### 2.3 GaussianNoise + ActionNoise (10 min)
- [ ] `core/augmentations/noise.py` — FrameTransform + TrajectoryTransform
- [ ] Unit test: verify noise applied, verify shape preserved

### 2.4 Mirror + AlohaAdapter (25 min) — KEY DIFFERENTIATOR
- [ ] `adapters/base.py` — ActionAdapter protocol
- [ ] `adapters/aloha.py` — AlohaAdapter (14-DOF, L/R swap, flip signs)
- [ ] `core/augmentations/mirror.py` — flip frames + swap/negate actions via adapter
- [ ] Unit test: flip frame, verify L/R actions swapped, verify sign negation
- [ ] **DEMO 2:** Load aloha → Mirror + ColorJitter → 100 episodes → push → visualizer link

---

## Phase 3: CLI + Polish (17:30 - 18:15, ~45 min)

**Goal:** Ship a proper CLI that makes the demo look professional.

### 3.1 CLI core (20 min)
- [ ] `cli/main.py` — Typer app with `augment` command
- [ ] Simple flag mode: --mirror, --color-jitter, --gaussian-noise, --action-noise
- [ ] --dataset, --output, --adapter, --multiply
- [ ] --no-upload flag
- [ ] Rich progress output

### 3.2 Config file support (10 min)
- [ ] `--config pipeline.yaml` parsing
- [ ] Load YAML → build Compose/OneOf/SomeOf pipeline
- [ ] Example config: `examples/aloha_basic.yaml`

### 3.3 Supporting commands (10 min)
- [ ] `robotq list` — show augmentations with type/adapter/deps
- [ ] `robotq preview` — single episode preview, save before/after PNGs
- [ ] `--dry-run` for augment command

### 3.4 README (5 min)
- [ ] What it does, how to install, how to run
- [ ] CLI examples + Python API example
- [ ] How AI coding agents were used
- [ ] **DEMO 3:** Full CLI demo from terminal

---

## Phase 4: Stretch Goals (18:15 - 19:00, ~45 min)

Pick based on time remaining. Ordered by demo impact:

### 4.1 SpeedWarp (15 min)
- [ ] `core/augmentations/speed.py` — resample episode at different tempo
- [ ] Unit test: verify frame count changes, actions interpolated

### 4.2 BackgroundReplace — generative hero (30 min, only if GPU available)
- [ ] `core/augmentations/background.py` — SAM2 + SD Inpainting
- [ ] `fast` method: frame differencing mask + inpainting
- [ ] Optional dependency gate: skip gracefully if torch/diffusers not installed
- [ ] **DEMO 4:** Same robot in different environments

### 4.3 MCP Server (15 min)
- [ ] `mcp/server.py` — expose augment_dataset, list_augmentations tools
- [ ] Test from Claude Code

### 4.4 Rust kernel (15 min, only if Python MVP is solid)
- [ ] `engine/` — PyO3 frame flip/color ops
- [ ] Benchmark vs numpy

---

## Decision Points

| Time | Check | Action if behind |
|------|-------|-----------------|
| 16:00 | Video decode/encode working? | If no: simplify to image-only dataset, skip video |
| 16:30 | Can push valid dataset to Hub? | If no: focus 100% on writer.py, skip augmentations |
| 17:30 | Mirror + pipeline working? | If no: ship ColorJitter-only demo, polish CLI |
| 18:15 | CLI working? | If no: skip stretch goals, ship what works |

## What "Done" Looks Like

Minimum: `robotq augment --dataset lerobot/aloha_static_cups_open --color-jitter --mirror --adapter aloha --output yourname/cups-aug` → valid dataset on Hub → visualizer link works → README explains what/how/why.
