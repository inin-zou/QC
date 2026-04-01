# RobotQ — Agentic Engineering Playbook

Concrete agent dispatch instructions for each phase. Claude Code should follow these literally.

---

## How to Use This Doc

Each phase has:
1. **Pre-requisites** — what must exist before dispatching
2. **Parallel batch** — agents to launch simultaneously (use `Agent` tool with multiple invocations in one message)
3. **Sequential follow-up** — agents that depend on the parallel batch
4. **Validation gate** — tests to run before moving to next phase
5. **Review gate** — code review after the phase

---

## Phase 0: Project Scaffold (do this yourself, no agents)

Before ANY agents, set up the skeleton:

```
1. uv init (if needed) + create pyproject.toml (requires-python >= 3.12)
2. Create directory structure:
   robotq/
   ├── __init__.py
   ├── core/__init__.py
   ├── core/augmentations/__init__.py
   ├── io/__init__.py
   ├── adapters/__init__.py
   ├── cli/__init__.py
   tests/
   ├── __init__.py
   ├── unit/__init__.py
   ├── integration/__init__.py
   ├── conftest.py
3. uv pip install -e ".[dev]"
4. Verify: python -c "import robotq; print('OK')"
```

---

## Phase 1: Foundation (dispatch 3 agents in parallel)

### Pre-requisites
- Phase 0 complete
- `uv pip install -e .` works

### Parallel Batch 1A — launch all 3 simultaneously

**Agent 1: Episode Container**
```
subagent_type: general-purpose
isolation: worktree
prompt: |
  Context: We're building RobotQ, a dataset augmentation tool for LeRobot v3 datasets.
  Working directory: /Users/yongkangzou/Desktop/ai projects/Qualia/QC

  Read these docs first:
  - .claude/docs/design-patterns.md (section 1: Episode as Universal Data Container)
  - .claude/docs/architecture.md (Episode section)

  Your task: Implement the Episode data container.

  Create file: robotq/core/episode.py
  - EpisodeMetadata dataclass with fields: episode_index (int), task_description (str),
    task_id (int), fps (float), camera_names (list[str]), robot_type (str), extra (dict[str, Any])
  - Episode dataclass with fields: frames (dict[str, list[np.ndarray]]),
    actions (np.ndarray), states (np.ndarray), metadata (EpisodeMetadata)
  - Add helper properties: num_frames (int), action_dim (int), state_dim (int)
  - Add validation in __post_init__: frames lengths match across cameras, match actions/states T dim

  Create file: tests/unit/test_episode.py
  - Test: create valid Episode, verify all fields accessible
  - Test: num_frames, action_dim, state_dim properties correct
  - Test: mismatched frame counts raises ValueError
  - Test: metadata extra field works

  Run: pytest tests/unit/test_episode.py -v
  All tests must pass before you're done.
  Do not modify any files outside your scope.
```

**Agent 2: Video I/O**
```
subagent_type: general-purpose
isolation: worktree
prompt: |
  Context: We're building RobotQ, a dataset augmentation tool for LeRobot v3 datasets.
  Working directory: /Users/yongkangzou/Desktop/ai projects/Qualia/QC

  Read these docs first:
  - .claude/docs/design-patterns.md (section 5: I/O Separation)
  - .claude/docs/architecture.md (I/O Layer section)

  Your task: Implement video decoding (decode only — LeRobot handles encoding).

  Create file: robotq/io/video.py
  One function only:
  - decode_video(path: str | Path) -> list[np.ndarray]
    Uses OpenCV VideoCapture. Returns list of (H, W, C) uint8 arrays in RGB order.
    Raises FileNotFoundError if path doesn't exist.
    Raises ValueError if video has 0 frames.

  This module does NOT encode video. LeRobot handles encoding via its dataset writer.
  This module knows NOTHING about datasets, episodes, or HuggingFace. Pure decoder.

  Create file: tests/unit/test_video.py
  - Test: create a short test video with OpenCV, decode it, verify frame count and shape
  - Test: decode nonexistent file -> FileNotFoundError
  - Test: frames are returned in RGB order (not BGR)

  Run: pytest tests/unit/test_video.py -v
  All tests must pass before you're done.
  Do not modify any files outside your scope.
```

**Agent 3: Schema Parsing**
```
subagent_type: general-purpose
isolation: worktree
prompt: |
  Context: We're building RobotQ, a dataset augmentation tool for LeRobot v3 datasets.
  Working directory: /Users/yongkangzou/Desktop/ai projects/Qualia/QC

  Read these docs first:
  - .claude/docs/design-patterns.md (section 5: I/O Separation)
  - .claude/docs/architecture.md (I/O Layer section)

  Your task: Implement LeRobot v3 schema parsing.

  First, research the LeRobot v3 dataset format. A LeRobot v3 dataset has:
  - meta/info.json: contains codebase_version, robot_type, fps, total_episodes,
    total_frames, features (dict of column definitions with dtype/shape), data_path template,
    video_path template, splits, chunks_size, etc.
  - meta/tasks.jsonl: one JSON object per line with task_index and task fields

  Create file: robotq/io/schema.py
  - parse_info(path: str | Path) -> dict: Read and return info.json as dict.
    Validate required keys exist: codebase_version, fps, total_episodes, features.
  - parse_tasks(path: str | Path) -> list[dict]: Read tasks.jsonl, return list of
    {"task_index": int, "task": str} dicts.
  - get_video_path(info: dict, episode_index: int, video_key: str) -> str:
    Use info["video_path"] template to construct path for a specific episode/camera.
  - get_data_path(info: dict, episode_index: int) -> str:
    Use info["data_path"] template to construct parquet path for a specific episode.
  - get_camera_names(info: dict) -> list[str]:
    Extract camera names from features dict (keys matching "observation.images.*").

  This module knows NOTHING about video decoding or episode objects. Pure JSON/JSONL parsing.

  Create file: tests/unit/test_schema.py
  - Test: parse valid info.json (create a fixture with minimal valid content)
  - Test: parse_info with missing required key -> KeyError or ValueError
  - Test: parse_tasks with valid JSONL
  - Test: get_video_path produces correct path from template
  - Test: get_data_path produces correct path from template
  - Test: get_camera_names extracts correct names

  Run: pytest tests/unit/test_schema.py -v
  All tests must pass before you're done.
  Do not modify any files outside your scope.
```

### After Parallel Batch 1A — validation gate

```bash
# Merge worktrees, then:
pytest tests/unit/test_episode.py tests/unit/test_video.py tests/unit/test_schema.py -v
python -c "
from robotq.core.episode import Episode, EpisodeMetadata
from robotq.io.video import decode_video
from robotq.io.schema import parse_info, parse_tasks
print('Phase 1A imports OK')
"
```

### Sequential Batch 1B — launch 2 agents in parallel (depend on 1A)

**Agent 4: Dataset Loader**
```
subagent_type: general-purpose
isolation: worktree
prompt: |
  Context: We're building RobotQ, a dataset augmentation tool for LeRobot v3 datasets.
  Working directory: /Users/yongkangzou/Desktop/ai projects/Qualia/QC

  Read these docs first:
  - .claude/docs/design-patterns.md (sections 1 and 5)
  - .claude/docs/architecture.md (I/O Layer section)
  - robotq/core/episode.py (understand Episode container)
  - robotq/io/video.py (understand decode_video interface)
  - robotq/io/schema.py (understand parse_info, get_video_path, get_data_path, get_camera_names)

  Your task: Implement the dataset loader that reads a LeRobot v3 dataset and yields Episode objects.

  Create file: robotq/io/loader.py
  - load_dataset(repo_id: str, *, max_episodes: int | None = None, local_dir: str | None = None) -> list[Episode]

  Implementation:
  1. Use huggingface_hub.snapshot_download(repo_id) to download dataset (or use local_dir)
  2. Use schema.parse_info() to read meta/info.json
  3. Use schema.parse_tasks() to read meta/tasks.jsonl
  4. For each episode (up to max_episodes):
     a. Read the parquet file using polars (schema.get_data_path for path)
     b. Extract action and state columns as numpy arrays
     c. For each camera (schema.get_camera_names):
        decode video using video.decode_video(schema.get_video_path(...))
     d. Build Episode object with correct metadata
  5. Return list of Episodes

  Important details:
  - LeRobot v3 parquet files have columns: action, observation.state, episode_index,
    frame_index, timestamp, index, task_index, next.done
  - action and observation.state are stored as list columns in parquet (each cell is a list of floats)
  - Multiple episodes may be in one parquet file — filter by episode_index
  - The video_path template uses {episode_chunk:03d} and {episode_index:06d} — chunk = episode_index // chunks_size

  Create file: tests/unit/test_loader.py
  - Test with a mock/minimal local dataset structure (create fixtures with small parquet + short video)
  - Test: loaded episode has correct frame count
  - Test: loaded episode has correct action_dim and state_dim
  - Test: metadata populated correctly (fps, camera_names, robot_type)
  - Test: max_episodes parameter limits output

  Run: pytest tests/unit/test_loader.py -v
  All tests must pass before you're done.
  Do not modify any files outside your scope (only create loader.py and test_loader.py).
```

**Agent 5: Dataset Writer**
```
subagent_type: general-purpose
isolation: worktree
prompt: |
  Context: We're building RobotQ, a dataset augmentation tool for LeRobot v3 datasets.
  Working directory: /Users/yongkangzou/Desktop/ai projects/Qualia/QC

  Read these docs first:
  - .claude/docs/design-patterns.md (sections 1 and 5)
  - .claude/docs/architecture.md (I/O Layer + Writer Strategy section)
  - robotq/core/episode.py (understand Episode container)

  Your task: Implement the dataset writer that takes Episode objects and creates a new LeRobot v3 dataset.

  Create file: robotq/io/writer.py
  - write_dataset(episodes: list[Episode], *, repo_id: str, local_only: bool = False) -> str

  Implementation:
  1. Use lerobot.datasets.lerobot_dataset.LeRobotDataset.create() to initialize a new dataset:
     - repo_id=repo_id, fps=episodes[0].metadata.fps
     - features dict built from Episode structure (observation.images.* for each camera,
       observation.state, action — match dtype and shape from the episode data)
  2. For each episode:
     a. For each frame index t:
        - Build frame dict with all camera images, state, action
        - dataset.add_frame(frame_dict)
     b. dataset.save_episode(task=episode.metadata.task_description)
  3. dataset.finalize()  # CRITICAL: must call to flush parquet footers
  4. If not local_only: dataset.push_to_hub()
  5. Return visualizer link:
     f"https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2F{repo_id.replace('/', '%2F')}%2Fepisode_0"

  Also create:
  - generate_visualizer_link(repo_id: str, episode: int = 0) -> str
    Helper to build the link.

  Create file: tests/unit/test_writer.py
  - Test: write_dataset with local_only=True creates correct directory structure
  - Test: generate_visualizer_link produces correct URL
  - Test: writing 2 episodes produces correct episode count in output
  (Use small synthetic Episode objects for tests, not real datasets)

  Run: pytest tests/unit/test_writer.py -v
  All tests must pass before you're done.
  Do not modify any files outside your scope (only create writer.py and test_writer.py).
```

### After Parallel Batch 1B — validation gate

```bash
pytest tests/unit/ -v
python -c "
from robotq.io.loader import load_dataset
from robotq.io.writer import write_dataset, generate_visualizer_link
print('Phase 1B imports OK')
"
```

### Sequential 1C — First augmentation + round-trip (do this yourself, not agents)

This must be done sequentially because it's the critical integration point:

```
1. Implement core/transform.py — base classes (SequenceTransform minimum)
2. Implement core/augmentations/color.py — ColorJitter
3. Write tests/unit/test_color.py, run and pass
4. Integration test: load aloha (2 episodes) → ColorJitter → write locally → verify
5. Push to Hub → click visualizer link → VERIFY IT WORKS
```

**This is DEMO 1. Do not proceed to Phase 2 until the visualizer link shows augmented data.**

### Phase 1 Review Gate

Launch code-reviewer agent:
```
subagent_type: superpowers:code-reviewer
prompt: |
  Review all code written in Phase 1 of RobotQ against the project's design patterns and architecture.

  Read these docs:
  - .claude/docs/design-patterns.md
  - .claude/docs/architecture.md

  Then review these files:
  - robotq/core/episode.py
  - robotq/io/video.py
  - robotq/io/schema.py
  - robotq/io/loader.py
  - robotq/io/writer.py
  - robotq/core/transform.py
  - robotq/core/augmentations/color.py

  Check:
  1. Is Episode the universal data container? No loose dicts?
  2. Are I/O modules respecting their boundaries? (video.py doesn't know about episodes, etc.)
  3. Is SequenceTransform base class implemented correctly? (get_params once, apply to all frames)
  4. Are there any circular imports?
  5. Any obvious bugs or missing error handling at boundaries?

  Report issues with file paths and line numbers. Only flag real problems.
```

---

## Phase 2: Core Augmentations + Mirror

### Pre-requisites
- Phase 1 complete and reviewed
- DEMO 1 works (visualizer link shows augmented data)

### Parallel Batch 2A — launch 4 agents simultaneously

**Agent 6: Transform Base Classes (full set)**
```
subagent_type: general-purpose
isolation: worktree
prompt: |
  Context: We're building RobotQ. Phase 1 is done. Now extending the transform system.
  Working directory: /Users/yongkangzou/Desktop/ai projects/Qualia/QC

  Read: .claude/docs/design-patterns.md (section 2: Transform Hierarchy)
  Read: robotq/core/transform.py (current state — may only have SequenceTransform)

  Your task: Complete ALL four transform base classes in robotq/core/transform.py.
  Keep what exists if it's correct. Add what's missing.

  Classes needed:
  - Transform (abstract base): __init__(self, p=1.0), apply(episode) -> Episode, __call__ = apply
  - FrameTransform(Transform): apply_to_frame(frame) -> frame. apply() maps over all frames/cameras with DIFFERENT random params per frame.
  - SequenceTransform(Transform): get_params(episode) -> dict, apply_to_frame(frame, params) -> frame. apply() calls get_params ONCE, maps with SAME params.
  - TrajectoryTransform(Transform): apply_to_episode(episode) -> Episode. apply() delegates directly.
  - RobotTransform(Transform): __init__(adapter, p=1.0), apply_to_episode(episode, adapter) -> Episode. apply() passes self.adapter.

  All apply() methods check random() < self.p before running. If p check fails, return episode unchanged.

  Update file: tests/unit/test_transform.py
  - Test each base class dispatch pattern with a concrete test subclass
  - Test p=0.0 returns unchanged
  - Test p=1.0 always transforms
  - Test FrameTransform gives different params per frame
  - Test SequenceTransform gives same params across all frames

  Run: pytest tests/unit/test_transform.py -v
```

**Agent 7: GaussianNoise + ActionNoise**
```
subagent_type: general-purpose
isolation: worktree
prompt: |
  Context: We're building RobotQ. Phase 1 is done.
  Working directory: /Users/yongkangzou/Desktop/ai projects/Qualia/QC

  Read: .claude/docs/design-patterns.md (sections 1, 2)
  Read: robotq/core/transform.py (understand FrameTransform and TrajectoryTransform)
  Read: robotq/core/episode.py (understand Episode)

  Your task: Implement noise augmentations.

  Create file: robotq/core/augmentations/noise.py
  - GaussianNoise(FrameTransform): adds per-pixel Gaussian noise to frames
    __init__(sigma: float = 0.02, p: float = 1.0)
    apply_to_frame: add N(0, sigma*255) noise, clip to [0,255], keep uint8
  - ActionNoise(TrajectoryTransform): adds Gaussian noise to action trajectories
    __init__(sigma: float = 0.01, p: float = 1.0)
    apply_to_episode: add N(0, sigma) to episode.actions, return new Episode with modified actions.
    Frames and states remain unchanged.

  Create file: tests/unit/test_noise.py
  - Test: GaussianNoise changes pixel values
  - Test: GaussianNoise with sigma=0 leaves frames approximately unchanged
  - Test: GaussianNoise output has same shape/dtype as input
  - Test: ActionNoise changes action values
  - Test: ActionNoise leaves frames and states untouched
  - Test: ActionNoise output actions have same shape

  Run: pytest tests/unit/test_noise.py -v
```

**Agent 8: Adapter Protocol + AlohaAdapter**
```
subagent_type: general-purpose
isolation: worktree
prompt: |
  Context: We're building RobotQ. Phase 1 is done.
  Working directory: /Users/yongkangzou/Desktop/ai projects/Qualia/QC

  Read: .claude/docs/design-patterns.md (section 3: ActionAdapter Protocol)
  Read: .claude/docs/architecture.md (Adapter section)

  Your task: Implement the adapter system for robot-specific action schema knowledge.

  Create file: robotq/adapters/base.py
  - ActionAdapter (Protocol class):
    - robot_type property -> str
    - get_left_slice() -> slice
    - get_right_slice() -> slice
    - get_flip_signs() -> np.ndarray  (shape: action_dim, values +1 or -1)
    - swap_arms(vec: np.ndarray) -> np.ndarray  (swap left/right slices)
    - swap_arms_batch(vecs: np.ndarray) -> np.ndarray  (swap for (T, dim) array)

  Create file: robotq/adapters/aloha.py
  - AlohaAdapter implements ActionAdapter for ALOHA bimanual robot:
    - robot_type = "aloha"
    - 14-DOF: indices 0-6 are left arm (waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate, gripper)
    - indices 7-13 are right arm (same joints)
    - get_left_slice() -> slice(0, 7)
    - get_right_slice() -> slice(7, 14)
    - get_flip_signs(): For horizontal mirror, the waist joint (index 0 and 7) should negate
      because waist rotation direction flips in mirror. Other joints may also need sign
      changes depending on axis convention. Start with negating indices 0 and 7 (waist joints).
      Document that this may need tuning per-task.
    - swap_arms(vec): create new array, copy left->right and right->left
    - swap_arms_batch(vecs): same but for (T, 14) arrays

  Create file: tests/unit/test_adapter.py
  - Test: AlohaAdapter.robot_type == "aloha"
  - Test: get_left_slice and get_right_slice return correct slices
  - Test: swap_arms correctly swaps a known vector (e.g., [1,2,3,4,5,6,7, 8,9,10,11,12,13,14] -> [8,9,...,14, 1,2,...,7])
  - Test: swap_arms_batch works for (T, 14) array
  - Test: get_flip_signs has correct shape (14,) and values are +1 or -1

  Run: pytest tests/unit/test_adapter.py -v
```

**Agent 9: Pipeline Composition**
```
subagent_type: general-purpose
isolation: worktree
prompt: |
  Context: We're building RobotQ. Phase 1 is done.
  Working directory: /Users/yongkangzou/Desktop/ai projects/Qualia/QC

  Read: .claude/docs/design-patterns.md (section 4: Pipeline Composition)
  Read: robotq/core/transform.py (understand Transform base class with apply() and p)

  Your task: Implement composable pipeline operators.

  Create file: robotq/core/pipeline.py
  - Compose(Transform): takes list of transforms, applies them sequentially.
    Each transform's own p controls whether it fires.
    Compose itself has a p that controls whether the entire chain runs.
    __init__(transforms: list[Transform], p: float = 1.0)
    apply(episode) -> Episode: iterate transforms, call each transform.apply(episode)

  - OneOf(Transform): randomly selects ONE transform to apply.
    __init__(transforms: list[Transform], p: float = 1.0)
    apply(episode) -> Episode: pick one at random, apply it.

  - SomeOf(Transform): randomly selects N transforms to apply.
    __init__(transforms: list[Transform], n: tuple[int, int] = (1, 2), p: float = 1.0)
    apply(episode) -> Episode: pick random N in range [n[0], n[1]], apply selected in order.

  All three are Transform subclasses so they can nest (Compose containing OneOf, etc).

  Create file: tests/unit/test_pipeline.py
  - Test: Compose applies all transforms in order
  - Test: Compose with p=0 returns unchanged
  - Test: OneOf picks exactly one (run 100 times, verify only one transform applied each time)
  - Test: SomeOf picks within range
  - Test: Nested composition (Compose containing OneOf) works
  - Test: Empty Compose returns unchanged
  Use mock transforms that track whether they were called.

  Run: pytest tests/unit/test_pipeline.py -v
```

### After Parallel Batch 2A — validation gate

```bash
pytest tests/unit/ -v
python -c "
from robotq.core.transform import FrameTransform, SequenceTransform, TrajectoryTransform, RobotTransform
from robotq.core.pipeline import Compose, OneOf, SomeOf
from robotq.core.augmentations.noise import GaussianNoise, ActionNoise
from robotq.adapters.aloha import AlohaAdapter
print('Phase 2A imports OK')
"
```

### Sequential 2B — Mirror (do this yourself)

Mirror is the key differentiator and requires careful integration of adapter + transform + video:

```
1. Implement core/augmentations/mirror.py — RobotTransform
   - Flip all camera frames horizontally (np.fliplr)
   - Swap left/right arm actions via adapter.swap_arms_batch()
   - Swap left/right arm states via adapter.swap_arms_batch()
   - Apply flip signs via adapter.get_flip_signs()
   - Requires adapter in __init__, raises ValueError if None
2. Write tests/unit/test_mirror.py, run and pass
3. Integration: load aloha → Compose([Mirror(AlohaAdapter()), ColorJitter()]) → write → push
4. VERIFY: visualizer shows flipped video + correct robot behavior
```

**This is DEMO 2. The mirrored robot should look physically correct in the visualizer.**

### Phase 2 Review Gate

Same pattern as Phase 1 — launch code-reviewer agent checking all Phase 2 files against design-patterns.md.

---

## Phase 3: CLI + Polish

### Pre-requisites
- Phase 2 complete, DEMO 2 works
- All unit tests pass

### Parallel Batch 3A — launch 3 agents simultaneously

**Agent 10: CLI augment command**
```
subagent_type: general-purpose
isolation: worktree
prompt: |
  Context: We're building RobotQ. Phases 1-2 are done. All core modules exist and work.
  Working directory: /Users/yongkangzou/Desktop/ai projects/Qualia/QC

  Read: .claude/docs/product-design.md (CLI Interface section — the full spec)
  Read: .claude/docs/architecture.md
  Read all files in robotq/ to understand the existing API

  Your task: Implement the CLI using Typer.

  Create file: robotq/cli/main.py
  - app = typer.Typer() with "robotq" as the program name
  - augment command with params:
    Input: --dataset (str, required), --output (str, required)
    Pipeline flags: --mirror, --color-jitter, --gaussian-noise, --action-noise
    Control: --adapter (str, default "aloha"), --multiply (int, default 1),
             --dry-run (bool), --no-upload (bool)
    UX: --verbose (bool)
  - Implementation:
    1. Load dataset via io.loader.load_dataset()
    2. Build pipeline from flags (each flag adds the transform to a Compose list)
    3. If --dry-run: print pipeline summary and exit
    4. Apply pipeline to each episode (multiply times if --multiply > 1)
    5. Write via io.writer.write_dataset()
    6. Print visualizer link
  - Use rich.console for progress output (Rich progress bar during processing)
  - list command: print table of augmentations with name, type, adapter requirement, deps

  Update pyproject.toml to add CLI entry point:
    [project.scripts]
    robotq = "robotq.cli.main:app"

  Test manually: robotq --help, robotq list, robotq augment --dry-run

  Do not modify core/ or io/ files.
```

**Agent 11: Config file parsing**
```
subagent_type: general-purpose
isolation: worktree
prompt: |
  Context: We're building RobotQ. Core pipeline and transforms exist.
  Working directory: /Users/yongkangzou/Desktop/ai projects/Qualia/QC

  Read: .claude/docs/design-patterns.md (section 6: Config-Driven Pipeline Construction)
  Read: .claude/docs/product-design.md (config YAML example)
  Read: robotq/core/pipeline.py, robotq/core/augmentations/*.py, robotq/adapters/*.py

  Your task: Implement YAML config parsing that builds pipelines.

  Create file: robotq/core/config.py
  - REGISTRY dict mapping transform names to classes
  - build_pipeline(config: dict, adapter=None) -> Compose
    Reads config["pipeline"] list, constructs transforms from type + kwargs.
    Handles nested OneOf/SomeOf (they have a "transforms" sub-key).
    RobotTransform classes get adapter injected.
  - load_config(path: str | Path) -> dict
    Read YAML file, return dict.

  Create file: examples/aloha_basic.yaml
  - dataset: lerobot/aloha_static_cups_open
  - adapter: aloha_bimanual
  - output: test/aloha-aug
  - multiply: 2
  - pipeline with Mirror, OneOf(ColorJitter, GaussianNoise), and any other available transforms

  Create file: tests/unit/test_config.py
  - Test: build_pipeline from simple config -> correct Compose
  - Test: OneOf in config -> builds OneOf
  - Test: SomeOf in config -> builds SomeOf with correct n
  - Test: unknown type name -> clear error
  - Test: load_config reads YAML correctly
  - Test: RobotTransform gets adapter injected

  Run: pytest tests/unit/test_config.py -v
```

**Agent 12: README**
```
subagent_type: general-purpose
isolation: worktree
prompt: |
  Context: We're building RobotQ, a dataset augmentation tool for LeRobot v3 datasets.
  Working directory: /Users/yongkangzou/Desktop/ai projects/Qualia/QC

  Read ALL docs in .claude/docs/ to understand the full product.
  Read the existing README.md (it's the challenge description — we need to REPLACE it).

  Your task: Write a professional README.md for RobotQ.

  Sections:
  1. Title + one-liner description
  2. Key Features (bullet list: composable pipeline, action-aware augmentation, adapter system, LeRobot v3 native, HF Hub upload)
  3. Installation: uv-based install instructions (basic + [generative] optional)
  4. Quick Start: CLI example (simple flag mode)
  5. Advanced Usage: --config pipeline.yaml with example YAML
  6. Python API: code example showing library usage
  7. Available Augmentations: table (name, type, description, adapter required?)
  8. Available Adapters: table
  9. Architecture: brief description of layered design (core -> io -> cli)
  10. How AI Coding Agents Were Used: describe the agentic workflow (parallel agents, test-first, review gates). This is required by the challenge.
  11. Roadmap: MCP server, Rust kernels, more adapters

  Keep it concise but professional. No emojis. Use code blocks for examples.
  The README should make someone want to try the tool.
```

### After Parallel Batch 3A — validation + integration

```bash
# Merge all worktrees
uv pip install -e .
pytest tests/ -v

# CLI smoke test
robotq --help
robotq list
robotq augment --dataset lerobot/aloha_static_cups_open --color-jitter --adapter aloha --output test/smoke --dry-run

# Config test
robotq augment --config examples/aloha_basic.yaml --dry-run
```

### Sequential 3B — preview + final demo

```
1. Add preview command to cli/main.py (save before/after PNGs to preview/)
2. Full demo run:
   robotq augment --config examples/aloha_basic.yaml
3. Click visualizer link, verify
```

**This is DEMO 3 — the final deliverable.**

### Phase 3 Review Gate

Final code review of entire codebase before submission.

---

## Phase 4: Stretch Goals (if time permits)

Each is a single agent, launched one at a time based on remaining time:

**SpeedWarp** → **BackgroundReplace** → **MCP Server** → **Rust kernel**

Only attempt if Phase 3 DEMO 3 works perfectly.

---

## Summary: Agent Count Per Phase

| Phase | Parallel Agents | Sequential Steps | Total Agents |
|-------|----------------|------------------|-------------|
| 0 | 0 (manual) | 1 | 0 |
| 1A | 3 (episode, video, schema) | 0 | 3 |
| 1B | 2 (loader, writer) | 0 | 2 |
| 1C | 0 | 1 (ColorJitter + integration) | 0 |
| 1 Review | 1 (code-reviewer) | 0 | 1 |
| 2A | 4 (transforms, noise, adapter, pipeline) | 0 | 4 |
| 2B | 0 | 1 (Mirror + integration) | 0 |
| 2 Review | 1 (code-reviewer) | 0 | 1 |
| 3A | 3 (CLI, config, README) | 0 | 3 |
| 3B | 0 | 1 (preview + final demo) | 0 |
| 3 Review | 1 (code-reviewer) | 0 | 1 |
| **Total** | **15 agents** | **4 manual steps** | |
