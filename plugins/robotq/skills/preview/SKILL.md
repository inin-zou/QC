---
name: robotq:preview
description: Preview augmentation effects on a single episode before running a full augmentation. Saves before/after PNGs and shows action diff stats.
---

# Preview Augmentation

Quickly check what an augmentation looks like on a single episode without processing the full dataset.

## CLI

```bash
robotq preview \
  --dataset lerobot/aloha_static_cups_open \
  --episode 0 \
  --mirror --color-jitter \
  --adapter aloha
```

This saves before/after PNGs to `preview/` and prints an action/state diff summary.

## Python API

```python
import copy
from robotq.io.loader import load_dataset
from robotq.core.augmentations.mirror import Mirror
from robotq.adapters.aloha import AlohaAdapter

episodes = load_dataset("lerobot/aloha_static_cups_open", max_episodes=1)
original = episodes[0]

mirror = Mirror(adapter=AlohaAdapter())
augmented = mirror(copy.deepcopy(original))

# Compare frames
import numpy as np
diff = np.mean(np.abs(original.actions - augmented.actions))
print(f"Action diff: {diff:.4f}")
```

## Output

Saves to `preview/` directory:
- `before_frame_000.png`, `after_frame_000.png`
- `before_frame_200.png`, `after_frame_200.png`
- `before_frame_399.png`, `after_frame_399.png`

Plus a summary table with frame counts and action/state diffs.
