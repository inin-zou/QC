# RobotQ — Architecture

## Project Structure

```
robotq/
├── core/
│   ├── episode.py              # Canonical Episode container
│   ├── pipeline.py             # Compose, OneOf, SomeOf orchestration
│   ├── transform.py            # Base classes: RobotTransform, SequenceTransform, etc.
│   └── augmentations/
│       ├── __init__.py
│       ├── mirror.py           # Flip video + swap L/R arm actions
│       ├── color.py            # Brightness, contrast, saturation, hue
│       ├── noise.py            # Gaussian noise on images and/or actions
│       ├── speed.py            # Time-stretch episodes
│       └── background.py       # SAM2 + SD Inpainting (optional dep)
│
├── io/
│   ├── __init__.py
│   ├── loader.py               # Locate dataset (Hub/local), parse structure, return episode refs
│   ├── writer.py               # Write parquet + metadata + video, push_to_hub
│   ├── video.py                # Decode MP4 -> frames. Decode only — LeRobot handles encoding.
│   └── schema.py               # Parse/generate info.json, tasks.jsonl, stats.json
│
├── adapters/
│   ├── __init__.py
│   ├── base.py                 # ActionAdapter protocol
│   └── aloha.py                # ALOHA bimanual (14-DOF) remap rules
│
├── cli/
│   ├── __init__.py
│   └── main.py                 # Typer app: augment, preview, list, adapters
│
├── engine/                     # Rust kernels (Phase 2)
│   ├── Cargo.toml
│   └── src/lib.rs
│
├── mcp/                        # MCP server (Phase 2)
│   └── server.py
│
├── pyproject.toml
└── README.md
```

## Core Abstractions

### Episode (core/episode.py)

The canonical data container. Every transform receives and returns an Episode.

```python
@dataclass
class EpisodeMetadata:
    episode_index: int
    task_description: str
    task_id: int
    fps: float
    camera_names: list[str]
    robot_type: str
    extra: dict[str, Any]           # Extensible for dataset-specific fields

@dataclass
class Episode:
    frames: dict[str, list[np.ndarray]]   # camera_name -> list of (H, W, C) arrays
    actions: np.ndarray                    # (T, action_dim)
    states: np.ndarray                     # (T, state_dim)
    metadata: EpisodeMetadata
```

**Why a unified object:** Avoids loose fields. SpeedWarp can update `metadata.fps`. Mirror can update `metadata.task_description` (e.g., "left" -> "right"). BackgroundReplace gets `metadata.camera_names` to know which views to process.

### Transform Hierarchy (core/transform.py)

Four base classes, each for a different augmentation pattern:

```python
class FrameTransform:
    """Per-frame independent. Each frame gets different random params.
    Use for: GaussianNoise, RandomErasing."""

    def apply_to_frame(self, frame: np.ndarray) -> np.ndarray: ...

    # Default: map over all frames in all cameras
    def apply(self, episode: Episode) -> Episode:
        for cam in episode.frames:
            episode.frames[cam] = [self.apply_to_frame(f) for f in episode.frames[cam]]
        return episode


class SequenceTransform:
    """Temporally consistent. Params sampled ONCE per episode, applied to all frames.
    Use for: ColorJitter, BackgroundReplace.
    Prevents video flickering."""

    def get_params(self, episode: Episode) -> dict: ...
    def apply_to_frame(self, frame: np.ndarray, params: dict) -> np.ndarray: ...

    # Default: sample params once, apply consistently
    def apply(self, episode: Episode) -> Episode:
        params = self.get_params(episode)
        for cam in episode.frames:
            episode.frames[cam] = [self.apply_to_frame(f, params) for f in episode.frames[cam]]
        return episode


class TrajectoryTransform:
    """Episode-level. Operates on entire episode at once.
    Use for: SpeedWarp, ActionNoise.
    May change episode length, FPS, or metadata."""

    def apply_to_episode(self, episode: Episode) -> Episode: ...

    def apply(self, episode: Episode) -> Episode:
        return self.apply_to_episode(episode)


class RobotTransform:
    """Paired image+action. Requires an ActionAdapter for schema awareness.
    Use for: Mirror."""

    def apply_to_episode(self, episode: Episode, adapter: ActionAdapter) -> Episode: ...

    def apply(self, episode: Episode) -> Episode:
        # adapter must be set at construction time
        return self.apply_to_episode(episode, self.adapter)
```

**Why four types:**
- `FrameTransform` vs `SequenceTransform` prevents video flickering
- `TrajectoryTransform` can change episode length (SpeedWarp resamples)
- `RobotTransform` makes the adapter dependency explicit at the type level

### Pipeline Composition (core/pipeline.py)

Borrowed from Albumentations, adapted for robotics:

```python
class Compose:
    """Apply transforms sequentially."""
    def __init__(self, transforms: list, p: float = 1.0): ...
    def apply(self, episode: Episode) -> Episode:
        for t in self.transforms:
            if random.random() < t.p:
                episode = t.apply(episode)
        return episode

class OneOf:
    """Randomly select ONE transform to apply."""
    def __init__(self, transforms: list, p: float = 1.0): ...

class SomeOf:
    """Randomly select N transforms to apply."""
    def __init__(self, transforms: list, n: tuple[int, int], p: float = 1.0): ...
```

### Adapter Protocol (adapters/base.py)

```python
class ActionAdapter(Protocol):
    """Encodes robot-specific action schema knowledge."""

    @property
    def robot_type(self) -> str: ...

    def get_left_slice(self) -> slice:
        """Slice for left arm joints in action vector."""
        ...

    def get_right_slice(self) -> slice:
        """Slice for right arm joints in action vector."""
        ...

    def get_flip_signs(self) -> np.ndarray:
        """Per-dimension sign multipliers for horizontal mirror.
        Shape: (action_dim,). Values: +1 (unchanged) or -1 (negate)."""
        ...

    def swap_arms(self, vec: np.ndarray) -> np.ndarray:
        """Swap left and right arm values in action/state vector."""
        ...
```

**AlohaAdapter** implements this for the 14-DOF bimanual ALOHA robot:
- Left arm: indices 0-6 (waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate, gripper)
- Right arm: indices 7-13 (same joints)
- Mirror: swap left/right slices + negate relevant axes

## I/O Layer

### Data Flow

```
                          ┌─────────────────┐
                          │   HF Hub / Local │
                          └────────┬────────┘
                                   │
                          ┌────────▼────────┐
                          │   loader.py      │
                          │  - locate dataset│
                          │  - parse schema  │
                          │  - yield episode │
                          │    references    │
                          └────────┬────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
              ┌─────▼─────┐ ┌─────▼─────┐ ┌─────▼─────┐
              │ video.py   │ │ schema.py  │ │ (parquet)  │
              │ decode MP4 │ │ parse meta │ │ read cols  │
              └─────┬─────┘ └─────┬─────┘ └─────┬─────┘
                    │              │              │
                    └──────────────┼──────────────┘
                                   │
                          ┌────────▼────────┐
                          │  Episode object  │
                          └────────┬────────┘
                                   │
                          ┌────────▼────────┐
                          │  pipeline.py     │
                          │  (transform      │
                          │   chain)         │
                          └────────┬────────┘
                                   │
                          ┌────────▼────────┐
                          │  Augmented       │
                          │  Episode object  │
                          └────────┬────────┘
                                   │
                          ┌────────▼────────┐
                          │  writer.py       │
                          │  - create dataset│
                          │  - add_frame()   │
                          │  - save_episode()│
                          │  - push_to_hub() │
                          └────────┬────────┘
                                   │
                          ┌────────▼────────┐
                          │  HF Hub +        │
                          │  Visualizer Link │
                          └─────────────────┘
```

### Module Responsibilities

| Module | Does | Does NOT |
|--------|------|----------|
| `loader.py` | Locate dataset on Hub/local, parse dataset structure, iterate episodes, read parquet columns | Decode video, write anything |
| `video.py` | Decode MP4 -> numpy frames (decode only; LeRobot handles encoding) | Know about datasets, episodes, or Hub; encode video |
| `schema.py` | Parse/generate info.json, tasks.jsonl, stats.json, validate schema | Read parquet or video |
| `writer.py` | Create new LeRobotDataset, add frames, save episodes, recompute stats, push_to_hub | Download or read source datasets |

### Writer Strategy

Uses the official LeRobot write API:

```python
dataset = LeRobotDataset.create(
    repo_id=output_repo,
    fps=source_fps,
    features=source_features,
)

for episode in augmented_episodes:
    for t in range(len(episode)):
        dataset.add_frame({
            "observation.images.cam_high": episode.frames["cam_high"][t],
            "observation.state": episode.states[t],
            "action": episode.actions[t],
            ...
        })
    dataset.save_episode()

dataset.finalize()       # Critical: flush parquet footers
dataset.push_to_hub()
```

This ensures the output is a valid LeRobot v3 dataset that works with the visualizer and training tools.

## Augmentation Implementation Notes

### Mirror (core/augmentations/mirror.py)

The most robotics-specific augmentation. Steps:
1. Flip all camera frames horizontally (`np.fliplr`)
2. Swap left/right arm actions via adapter (`adapter.swap_arms(action)`)
3. Swap left/right arm states via adapter
4. Negate mirror-sensitive axes via adapter (`action * adapter.get_flip_signs()`)
5. Optionally update task description ("left" <-> "right")

**Requires adapter.** Will raise `ValueError` if no adapter is provided.

### ColorJitter (core/augmentations/color.py)

SequenceTransform — params sampled once per episode for temporal consistency.
- Brightness: factor in [1-strength, 1+strength]
- Contrast: same range
- Saturation: same range
- Hue: shift in [-strength, +strength]

### SpeedWarp (core/augmentations/speed.py)

TrajectoryTransform — resamples entire episode.
- Sample speed factor from [min_speed, max_speed]
- Resample frames via frame index interpolation (nearest or linear)
- Interpolate actions and states (linear)
- Update `metadata.fps` (or keep fps and change frame count)
- Preserve first and last frames (episode boundaries)
- Recompute timestamps

### BackgroundReplace (core/augmentations/background.py)

SequenceTransform — consistent background across episode.
- **auto method:** SAM2 segments robot+objects in first frame, propagates mask across episode, SD Inpainting fills masked background with text prompt
- **fast method:** Frame differencing (median background subtraction) generates mask, same inpainting step
- Params sampled once: the generated background is consistent across all frames in the episode
- Only processes specified cameras (or all by default)

## Rust Integration (Phase 2)

### Where Rust helps

| Hot path | Current (Python) | Rust kernel |
|----------|------------------|-------------|
| Frame horizontal flip | `np.fliplr` per frame | Batch flip in parallel |
| Color jitter application | PIL/OpenCV per frame | SIMD-accelerated pixel ops |
| Video frame iteration | OpenCV VideoCapture | Direct MP4 demuxing |
| Batch episode processing | multiprocessing | Rayon thread pool |

### Architecture

```
robotq/engine/
├── Cargo.toml              # PyO3 + image + rayon deps
└── src/
    ├── lib.rs              # PyO3 module definition
    ├── transforms.rs       # Frame-level transform kernels
    └── batch.rs            # Parallel episode processing
```

Python calls: `from robotq.engine import batch_flip_frames, batch_color_jitter`

Falls back to numpy/OpenCV if Rust extension not compiled.

### Build

```toml
# pyproject.toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
requires-python = ">=3.12"

[project.optional-dependencies]
generative = ["segment-anything-2", "diffusers", "torch"]
dev = ["pytest", "ruff"]
```

Rust kernels (Phase 2 stretch) would be a separate crate built via maturin, imported as an optional dependency.

## MCP Server (Phase 2)

### Tools exposed

```python
@mcp.tool()
def augment_dataset(
    source: str,            # HF repo ID
    output: str,            # Output repo ID
    augmentations: list,    # ["mirror", "color_jitter", ...]
    adapter: str = "auto",  # Robot adapter name
    multiply: int = 2,      # Dataset multiplication factor
) -> dict:
    """Augment a LeRobot dataset and upload to HF Hub."""
    ...

@mcp.tool()
def list_augmentations() -> list[dict]:
    """List available augmentation transforms."""
    ...

@mcp.tool()
def preview_augmentation(
    source: str,
    episode: int,
    augmentation: str,
) -> str:
    """Preview augmentation on a single episode. Returns path to preview."""
    ...
```

### Integration

The MCP server imports the same `core/` and `io/` modules as the CLI. No code duplication. It's a thin translation layer from MCP tool calls to library functions.

## Dependencies

### Core (managed by uv)
```
python >= 3.12
lerobot >= 0.4.0
polars
numpy
opencv-python-headless   # Decode only
typer
rich
huggingface_hub
pyyaml
```

### Optional (generative)
```
torch
diffusers
segment-anything-2
```

### Dev
```
pytest
ruff
```
