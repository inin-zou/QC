---
name: robotq:configure
description: Generate a YAML pipeline config file for RobotQ augmentation. Use when the user wants to create, edit, or customize a pipeline configuration.
---

# Generate Pipeline Config

Create a YAML config file for complex augmentation pipelines with Compose/OneOf/SomeOf and per-transform probability control.

## Config Format

```yaml
dataset: lerobot/aloha_static_cups_open
adapter: aloha          # 'aloha' for bimanual, 'generic' for single-arm
output: USERNAME/output  # HF Hub repo ID
multiply: 2              # Augmented copies per original

pipeline:
  - type: Mirror
    p: 0.5              # 50% chance per episode

  - type: OneOf          # Randomly pick ONE
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
    p: 0.3
```

## Available Transform Types

For `type` field in pipeline items:

| Type | Parameters | Notes |
|------|-----------|-------|
| `Mirror` | `p` | Requires adapter in config |
| `ColorJitter` | `brightness`, `contrast`, `saturation`, `hue`, `p` | All params in [0, 1], hue in [0, 0.5] |
| `GaussianNoise` | `sigma`, `p` | sigma scales pixel noise (0.02 = moderate) |
| `ActionNoise` | `sigma`, `p` | sigma scales action perturbation (0.01 = small) |
| `SpeedWarp` | `min_rate`, `max_rate`, `p` | rate < 1 = slower (more frames), > 1 = faster |
| `BackgroundReplace` | `prompt`, `method`, `strength`, `p` | method: 'fast' or 'auto' |

## Composition Operators

| Operator | What it does | Config |
|----------|-------------|--------|
| `OneOf` | Pick ONE random transform | `transforms:` list + `p` |
| `SomeOf` | Pick N random transforms | `transforms:` list + `n: [min, max]` + `p` |

## Usage

Save as `pipeline.yaml` then run:
```bash
robotq augment --config pipeline.yaml
```

Or preview first:
```bash
robotq augment --config pipeline.yaml --dry-run
```
