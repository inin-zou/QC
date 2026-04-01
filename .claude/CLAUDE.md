# CLAUDE.md — RobotQ Project Guide

## What This Project Is

A coding challenge for Qualia: build a **dataset augmentation tool for LeRobot v3 datasets** that automatically uploads augmented data to Hugging Face Hub.

**Product name:** RobotQ
**One-liner:** Composable augmentation toolkit for LeRobot v3 datasets — CLI-first, MCP-ready, with generative superpowers.

## The Assignment

- Build a tool that augments LeRobot v3 datasets (parquet + meta/ + videos/)
- Automatically upload the augmented dataset to HF Hub
- Print a visualizer link: `https://huggingface.co/spaces/lerobot/visualize_dataset?path=...`
- Use AI coding agents heavily (and document how)
- Time budget: 6 hours (13:00 - 19:00)
- Deliverable: public GitHub repo with working code + README

## What "Done" Looks Like

```bash
robotq augment --dataset lerobot/aloha_static_cups_open \
  --mirror --color-jitter --adapter aloha --multiply 2 \
  --output yourname/aloha-robotq-demo
```
→ Valid augmented dataset on HF Hub → Visualizer link works → README explains what/how/why.

## Docs Map

All design docs live in `.claude/docs/`. Read them in this order:

| Doc | What it covers | When to read |
|-----|---------------|--------------|
| **product-design.md** | Product identity, augmentation catalog, CLI interface spec, demo story, competitive positioning | Read first. This is the "what" and "why". |
| **architecture.md** | Project structure, Episode/Transform/Pipeline/Adapter abstractions, I/O data flow, Rust plan, MCP plan | Read second. This is the "how". |
| **design-patterns.md** | 8 patterns every module must follow: Episode container, Transform hierarchy, Adapter protocol, Pipeline composition, I/O separation, config-driven construction, error handling, naming | Read before writing ANY code. This is the "rules". |
| **ROADMAP.md** | 4-phase build plan with time estimates, decision points, and scope-cut triggers | Read to know what to build next and when to stop. |
| **agentic-engineering.md** | Concrete agent dispatch blocks for each phase — exact prompts, parallel batches, validation gates, review gates | Read when dispatching agents. Copy-paste the prompts literally. |
| **test.md** | Test structure, fixtures, unit test specs per module, integration tests, 9-step regression walkthrough | Read when writing tests or before demo/submission. |

## Key Technical Decisions

- **Python 3.12+** (matches LeRobot's requires-python)
- **uv** for dependency management
- **Polars** for parquet reading
- **OpenCV** for video decoding only — NO video encoding. LeRobot handles encoding internally.
- **LeRobotDataset.create() → add_frame() → save_episode() → push_to_hub()** for ALL dataset writing. No raw parquet/video writing. This guarantees valid output.
- **Typer + Rich** for CLI
- **Ruff** for lint + format
- **pytest** for testing
- **HF auth**: default to local login, support `--token` override
- Augmentations are **paired**: image transforms that affect geometry (Mirror) must also transform actions/states via an adapter

## Build Order (demo-first)

```
Phase 0: Scaffold (pyproject.toml, directory structure, pip install -e .)
Phase 1: Foundation + First Demo (video I/O, loader, writer, ColorJitter → push → visualizer works)
Phase 2: Core Augmentations (Mirror + AlohaAdapter, pipeline composition, noise transforms)
Phase 3: CLI + Polish (Typer CLI, --config YAML, preview, list, README)
Phase 4: Stretch (SpeedWarp, BackgroundReplace, MCP server, Rust kernels)
```

**Rule: each phase ends with a working demo. Do not start Phase N+1 until Phase N's demo works.**

## How to Work With Agents

Follow `.claude/docs/agentic-engineering.md` for exact agent dispatch instructions. The pattern:

1. **Dispatch parallel agents** for independent modules (e.g., video.py, schema.py, episode.py simultaneously)
2. **Merge and validate** — run `pytest tests/unit/ -v` + import smoke test
3. **Build dependent modules** sequentially or in the next parallel batch
4. **Integration test** at phase boundary — run the full pipeline end-to-end
5. **Code review** — launch code-reviewer agent to check against design-patterns.md
6. **Demo checkpoint** — verify the visualizer link works

## Rules

- **Episode is the universal data container.** All transforms receive and return Episode. No loose dicts.
- **Four transform base classes.** FrameTransform (per-frame random), SequenceTransform (consistent per episode), TrajectoryTransform (episode-level), RobotTransform (paired image+action via adapter). Pick the right one.
- **Adapters encode robot knowledge.** Mirror needs an adapter. ColorJitter doesn't. Never hardcode action schemas in transforms.
- **I/O modules stay in their lanes.** video.py is a pure codec. loader.py yields Episodes. writer.py consumes Episodes. schema.py parses JSON. No crossover.
- **Test immediately.** Write unit tests right after each module. Run them. Green before moving on.
- **Config = code.** YAML config maps 1:1 to Python objects via a registry. Every transform is constructable from keyword args.

## Demo Dataset

`lerobot/aloha_static_cups_open` — 50 episodes, 4 cameras (cam_high, cam_left_wrist, cam_low, cam_right_wrist), 14-DOF bimanual ALOHA robot, 50 FPS, ~400 frames per episode.

## Dependencies (managed by uv)

```
# Core
python >= 3.12
lerobot >= 0.4.0
polars
numpy
opencv-python-headless
typer
rich
huggingface_hub
pyyaml

# Optional (generative)
torch
diffusers
segment-anything-2

# Dev
pytest
ruff
```
