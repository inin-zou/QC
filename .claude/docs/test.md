# RobotQ — Test Guide & Regression Suite

Step-by-step test plan to verify the product actually works. Run through this before any demo or submission.

---

## Test Structure

```
tests/
├── unit/
│   ├── test_episode.py          # Episode container
│   ├── test_video.py            # MP4 decode/encode
│   ├── test_schema.py           # info.json/tasks.jsonl parsing
│   ├── test_loader.py           # Dataset loading
│   ├── test_writer.py           # Dataset writing
│   ├── test_transform.py        # Base class dispatch
│   ├── test_pipeline.py         # Compose/OneOf/SomeOf
│   ├── test_color.py            # ColorJitter
│   ├── test_noise.py            # GaussianNoise + ActionNoise
│   ├── test_mirror.py           # Mirror + AlohaAdapter
│   ├── test_speed.py            # SpeedWarp (if implemented)
│   └── test_config.py           # YAML config parsing
├── integration/
│   ├── test_roundtrip.py        # Load → write → load → compare
│   ├── test_augment_pipeline.py # Full pipeline on real data
│   └── test_hub_upload.py       # Push to Hub + verify link
└── conftest.py                  # Shared fixtures (sample episodes, temp dirs)
```

---

## Fixtures (conftest.py)

```python
@pytest.fixture
def sample_episode():
    """Minimal valid Episode for unit tests. No real data needed."""
    return Episode(
        frames={"cam_high": [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(10)]},
        actions=np.random.randn(10, 14).astype(np.float32),
        states=np.random.randn(10, 14).astype(np.float32),
        metadata=EpisodeMetadata(
            episode_index=0,
            task_description="open cups",
            task_id=0,
            fps=50.0,
            camera_names=["cam_high"],
            robot_type="aloha",
        ),
    )

@pytest.fixture
def aloha_adapter():
    return AlohaAdapter()

@pytest.fixture
def real_dataset_id():
    """Real dataset for integration tests. Requires network."""
    return "lerobot/aloha_static_cups_open"
```

---

## Unit Tests — What Each Must Verify

### test_episode.py
- [ ] Create Episode with valid data → no error
- [ ] Access frames by camera name → correct shape
- [ ] Access actions/states → correct shape and dtype
- [ ] Metadata fields accessible

### test_video.py
- [ ] Decode MP4 → returns list of numpy arrays with correct shape
- [ ] Encode frames → MP4 → file exists and is valid
- [ ] Round-trip: decode → encode → decode → frame count matches
- [ ] Handles different resolutions (480x640, 240x320)

### test_schema.py
- [ ] Parse valid info.json → correct fields
- [ ] Parse tasks.jsonl → correct task descriptions
- [ ] Generate info.json from Episode metadata → valid JSON

### test_loader.py
- [ ] Load dataset from Hub → returns Episodes
- [ ] Episode has correct number of frames, action dim, state dim
- [ ] Metadata populated (fps, camera_names, robot_type)
- [ ] Can load specific episodes by index

### test_writer.py
- [ ] Write Episode to local path → valid LeRobot v3 structure
- [ ] Parquet files created with correct columns
- [ ] Video files created and playable
- [ ] info.json and tasks.jsonl generated correctly

### test_transform.py
- [ ] FrameTransform.apply() maps over all frames in all cameras
- [ ] SequenceTransform.apply() calls get_params() once, applies to all frames
- [ ] TrajectoryTransform.apply() delegates to apply_to_episode()
- [ ] RobotTransform.apply() passes adapter
- [ ] Probability p=0.0 → episode unchanged
- [ ] Probability p=1.0 → episode always transformed

### test_pipeline.py
- [ ] Compose applies transforms in order
- [ ] OneOf picks exactly one transform
- [ ] SomeOf picks N transforms within range
- [ ] Nested composition works (Compose containing OneOf)
- [ ] Empty Compose returns episode unchanged
- [ ] Probability on composition groups works

### test_color.py
- [ ] ColorJitter changes pixel values
- [ ] Actions and states remain unchanged
- [ ] All frames in episode have same jitter params (temporal consistency)
- [ ] Brightness=0, contrast=0, sat=0, hue=0 → frames approximately unchanged
- [ ] Output frames have same shape and dtype as input

### test_noise.py
- [ ] GaussianNoise changes pixel values
- [ ] Different frames get different noise (FrameTransform, not Sequence)
- [ ] ActionNoise changes action values
- [ ] ActionNoise leaves frames and states unchanged
- [ ] Noise magnitude scales with sigma parameter

### test_mirror.py
- [ ] Frames are horizontally flipped (left pixel = right pixel of original)
- [ ] Left arm actions move to right arm slice position
- [ ] Right arm actions move to left arm slice position
- [ ] Flip signs applied correctly (specific axes negated)
- [ ] States also swapped via adapter
- [ ] Raises ValueError if no adapter provided

### test_config.py
- [ ] Parse simple YAML → builds correct Compose pipeline
- [ ] Parse OneOf in YAML → builds OneOf with correct children
- [ ] Parse SomeOf in YAML → builds SomeOf with correct n range
- [ ] adapter field resolved correctly
- [ ] Invalid type name → clear error message

---

## Integration Tests — End-to-End Verification

### test_roundtrip.py
```
Load aloha episode 0 → Write to temp dir → Load from temp dir → Compare
- Frame count matches
- Action values match (within float tolerance)
- State values match
- Metadata matches
```

### test_augment_pipeline.py
```
Load aloha → Build pipeline (Mirror + ColorJitter) → Augment all episodes → Write → Load augmented
- Episode count doubled (if multiply=2)
- Augmented frames have different pixels than originals
- Mirrored episodes have swapped L/R actions
- Original episodes preserved unchanged
```

### test_hub_upload.py (manual / CI-only)
```
Load aloha → ColorJitter → Push to Hub → Fetch dataset from Hub → Verify
- Dataset exists on Hub
- Visualizer link returns 200
- Episode count correct
```

---

## Regression Test Walkthrough

Run this before demo or submission. Each step must pass before proceeding.

### Step 1: Unit tests (all green)
```bash
pytest tests/unit/ -v
```

### Step 2: Import smoke test
```bash
python -c "
from robotq.core.episode import Episode, EpisodeMetadata
from robotq.core.pipeline import Compose, OneOf
from robotq.core.augmentations.color import ColorJitter
from robotq.core.augmentations.noise import GaussianNoise, ActionNoise
from robotq.core.augmentations.mirror import Mirror
from robotq.adapters.aloha import AlohaAdapter
from robotq.io.loader import load_dataset
from robotq.io.writer import write_dataset
print('All imports OK')
"
```

### Step 3: Load real dataset
```bash
python -c "
from robotq.io.loader import load_dataset
episodes = list(load_dataset('lerobot/aloha_static_cups_open', max_episodes=2))
print(f'Loaded {len(episodes)} episodes')
ep = episodes[0]
print(f'Frames: {len(ep.frames[ep.metadata.camera_names[0]])}')
print(f'Actions shape: {ep.actions.shape}')
print(f'FPS: {ep.metadata.fps}')
"
```

### Step 4: Augment + write locally
```bash
python -c "
from robotq.io.loader import load_dataset
from robotq.io.writer import write_dataset
from robotq.core.pipeline import Compose
from robotq.core.augmentations.color import ColorJitter

episodes = list(load_dataset('lerobot/aloha_static_cups_open', max_episodes=2))
pipeline = Compose([ColorJitter(brightness=0.3)])
augmented = [pipeline(ep) for ep in episodes]
write_dataset(augmented, repo_id='test-local', local_only=True)
print('Local write OK')
"
```

### Step 5: CLI dry-run
```bash
robotq augment \
  --dataset lerobot/aloha_static_cups_open \
  --color-jitter \
  --adapter aloha \
  --output yourname/test-aug \
  --dry-run
```

### Step 6: CLI full run (the real demo)
```bash
robotq augment \
  --dataset lerobot/aloha_static_cups_open \
  --color-jitter \
  --mirror \
  --adapter aloha \
  --multiply 2 \
  --output yourname/aloha-robotq-demo
```

### Step 7: Verify visualizer link
- Click the printed link
- Verify augmented episodes are visible
- Check that mirrored episodes show flipped video
- Verify episode count matches expectation

### Step 8: Config file mode
```bash
robotq augment --config examples/aloha_basic.yaml
```

### Step 9: Preview command
```bash
robotq preview \
  --dataset lerobot/aloha_static_cups_open \
  --episode 0 \
  --mirror \
  --adapter aloha
# Check preview/ directory for before/after PNGs
```

---

## Quick Smoke Test (30 seconds)

For rapid validation during development:

```bash
pytest tests/unit/ -v --tb=short && \
python -c "from robotq.core.pipeline import Compose; print('OK')"
```

---

## Known Edge Cases to Test

- [ ] Dataset with only 1 episode
- [ ] Dataset with no video (image-only)
- [ ] Episode with 0 frames (should error gracefully)
- [ ] Mirror with no adapter (should raise ValueError)
- [ ] SpeedWarp with rate=1.0 (should return unchanged)
- [ ] ColorJitter with all params=0 (should return approximately unchanged)
- [ ] Pipeline with p=0 for all transforms (should return unchanged)
- [ ] Very short episode (< 5 frames)
- [ ] HF Hub upload with invalid token (should error with clear message)
