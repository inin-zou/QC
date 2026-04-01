# RobotQ — Design Patterns & Reusable Interfaces

Quick reference for patterns that MUST be followed consistently across the codebase. Every new module should check this doc before writing code.

---

## 1. Episode as the Universal Data Container

**Pattern:** All data flows through `Episode`. No loose dicts, no positional args.

```python
@dataclass
class EpisodeMetadata:
    episode_index: int
    task_description: str
    task_id: int
    fps: float
    camera_names: list[str]
    robot_type: str
    extra: dict[str, Any] = field(default_factory=dict)

@dataclass
class Episode:
    frames: dict[str, list[np.ndarray]]   # camera_name -> [(H,W,C), ...]
    actions: np.ndarray                    # (T, action_dim)
    states: np.ndarray                     # (T, state_dim)
    metadata: EpisodeMetadata
```

**Rules:**
- Every transform takes `Episode` → returns `Episode`
- Never pass `frames, actions, states` as separate args
- Metadata updates (fps, task_description) happen inside the transform, not outside
- `Episode` is immutable-ish: transforms should create new arrays, not mutate in place (unless perf requires it, then document)

---

## 2. Transform Hierarchy — Four Base Classes

**Pattern:** Every augmentation inherits from exactly one base class. The base class defines the dispatch pattern.

```
Transform (abstract)
├── FrameTransform        — per-frame independent, each frame gets different random params
├── SequenceTransform     — temporally consistent, params sampled ONCE per episode
├── TrajectoryTransform   — episode-level, may change length/fps/metadata
└── RobotTransform        — paired image+action, requires ActionAdapter
```

**Interface contract:**

```python
class Transform:
    def __init__(self, p: float = 1.0): ...
    def apply(self, episode: Episode) -> Episode: ...
    # Each subclass provides default apply() that dispatches to specific methods

class FrameTransform(Transform):
    def apply_to_frame(self, frame: np.ndarray) -> np.ndarray: ...
    # apply() maps over all frames in all cameras

class SequenceTransform(Transform):
    def get_params(self, episode: Episode) -> dict: ...
    def apply_to_frame(self, frame: np.ndarray, params: dict) -> np.ndarray: ...
    # apply() calls get_params() once, then maps apply_to_frame() with same params

class TrajectoryTransform(Transform):
    def apply_to_episode(self, episode: Episode) -> Episode: ...
    # apply() delegates directly

class RobotTransform(Transform):
    def __init__(self, adapter: ActionAdapter, p: float = 1.0): ...
    def apply_to_episode(self, episode: Episode, adapter: ActionAdapter) -> Episode: ...
    # apply() passes self.adapter
```

**Rules:**
- ALWAYS implement the specific methods (`apply_to_frame`, `get_params`, etc.), not `apply()` directly
- `apply()` is the public API; specific methods are the extension points
- Probability `p` is checked in `apply()` before dispatching — subclasses don't check it
- `get_params()` in SequenceTransform must be deterministic given a seed — for reproducibility

---

## 3. ActionAdapter Protocol

**Pattern:** Robot-specific knowledge is encoded in adapters, never hardcoded in transforms.

```python
class ActionAdapter(Protocol):
    @property
    def robot_type(self) -> str: ...
    def get_left_slice(self) -> slice: ...
    def get_right_slice(self) -> slice: ...
    def get_flip_signs(self) -> np.ndarray: ...
    def swap_arms(self, vec: np.ndarray) -> np.ndarray: ...
```

**Rules:**
- Transforms that need schema knowledge take `adapter` in `__init__`, not per-call
- Transforms that DON'T need schema knowledge must NOT accept an adapter
- New robot support = new adapter file in `adapters/`, zero changes to transforms
- Adapter methods return numpy arrays/slices — no framework-specific types

---

## 4. Pipeline Composition — Albumentations Pattern

**Pattern:** Pipelines are trees of transforms, not flat lists.

```python
pipeline = Compose([
    Mirror(adapter=AlohaAdapter(), p=0.5),
    OneOf([
        ColorJitter(brightness=0.3),
        GaussianNoise(sigma=0.02),
    ], p=0.8),
    SpeedWarp(min_rate=0.8, max_rate=1.2),
])

augmented_episode = pipeline(episode)  # __call__ delegates to apply()
```

**Composition operators:**
- `Compose` — sequential, all transforms applied in order (each with its own `p`)
- `OneOf` — randomly select ONE transform to apply
- `SomeOf(n=(1,3))` — randomly select N transforms to apply

**Rules:**
- `Compose`, `OneOf`, `SomeOf` are themselves `Transform` subclasses — they nest
- Pipeline `__call__` is an alias for `apply()`
- Pipeline must be serializable to/from YAML (for `--config` support)
- Each composition operator has its own `p` — the probability the entire group fires

---

## 5. I/O Separation — Loader / Writer / Video / Schema

**Pattern:** Four modules with non-overlapping responsibilities.

| Module | Input | Output | Never does |
|--------|-------|--------|------------|
| `loader.py` | HF repo ID or local path | Iterator of `Episode` objects | Write files |
| `writer.py` | Iterator of `Episode` objects + output repo | Dataset on Hub + visualizer link | Read source data, decode video |
| `video.py` | MP4 file path | `list[np.ndarray]` | Encode video, know about datasets or episodes |
| `schema.py` | JSON/JSONL file paths | Parsed metadata dicts | Read parquet or video |

**Rules:**
- `video.py` is decode-only — it takes MP4 paths and returns frames. LeRobot handles all encoding via `LeRobotDataset.create() → add_frame() → save_episode()`.
- `loader.py` uses `video.py` and `schema.py` internally but presents `Episode` objects externally
- `writer.py` uses the LeRobot `LeRobotDataset` API. It does NOT use `video.py` — LeRobot encodes video internally.
- No circular imports between io modules

---

## 6. Config-Driven Pipeline Construction

**Pattern:** YAML config maps 1:1 to Python objects via a registry.

```yaml
pipeline:
  - type: Mirror
    p: 0.5
  - type: ColorJitter
    brightness: 0.3
```

```python
REGISTRY = {
    "Mirror": Mirror,
    "ColorJitter": ColorJitter,
    "GaussianNoise": GaussianNoise,
    ...
}

def build_pipeline(config: dict, adapter=None) -> Compose:
    transforms = []
    for item in config["pipeline"]:
        cls = REGISTRY[item.pop("type")]
        if issubclass(cls, RobotTransform):
            transforms.append(cls(adapter=adapter, **item))
        else:
            transforms.append(cls(**item))
    return Compose(transforms)
```

**Rules:**
- Every transform must be constructable from keyword args only (no positional-only params)
- The registry is the single source of truth for available transforms
- `OneOf` and `SomeOf` configs nest via a `transforms` key
- Adapter is resolved at pipeline build time from `--adapter` flag or config `adapter` field

---

## 7. Error Handling Philosophy

- **Fail fast at boundaries** — validate dataset exists, adapter is compatible, deps installed BEFORE processing
- **No silent fallbacks** — if SAM2 isn't installed, raise `ImportError` with install instructions, don't silently skip
- **Transforms are trusted** — no try/catch inside transform chains. If a transform crashes, it's a bug.
- **Writer validates output** — after `finalize()`, verify episode count matches expectation

---

## 8. Naming Conventions

| Thing | Convention | Example |
|-------|-----------|---------|
| Transform class | PascalCase, descriptive | `ColorJitter`, `BackgroundReplace` |
| Adapter class | PascalCase + "Adapter" | `AlohaAdapter` |
| Module files | snake_case, singular | `mirror.py`, `color.py` |
| Config keys | snake_case | `min_rate`, `brightness` |
| CLI flags | kebab-case | `--color-jitter`, `--speed-warp` |
| Test files | `test_` prefix | `test_mirror.py`, `test_pipeline.py` |
