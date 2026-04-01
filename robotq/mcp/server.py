"""RobotQ MCP server — exposes dataset augmentation as tools for AI agents.

Thin MCP (Model Context Protocol) server that wraps RobotQ's augmentation
pipeline so that AI coding assistants (Claude Code, Cursor, etc.) can call
augmentation operations directly.

Run as a module::

    python -m robotq.mcp.server
"""

from __future__ import annotations

import copy
import logging
from typing import Optional

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    raise ImportError(
        "MCP server requires the 'mcp' package. Install with: uv pip install 'mcp[cli]'"
    )

logger = logging.getLogger(__name__)

mcp = FastMCP("robotq", instructions="Augment LeRobot v3 datasets with composable transforms")

# ---------------------------------------------------------------------------
# Name -> class mapping (lowercase convenience names used by MCP tools)
# ---------------------------------------------------------------------------

_AUGMENTATION_NAME_MAP: dict[str, str] = {
    "mirror": "Mirror",
    "color_jitter": "ColorJitter",
    "gaussian_noise": "GaussianNoise",
    "action_noise": "ActionNoise",
    "speed_warp": "SpeedWarp",
}


def _resolve_augmentation_class(name: str):
    """Import and return an augmentation class by its short name."""
    canon = _AUGMENTATION_NAME_MAP.get(name)
    if canon is None:
        raise ValueError(
            f"Unknown augmentation: {name!r}. Available: {sorted(_AUGMENTATION_NAME_MAP.keys())}"
        )

    if canon == "Mirror":
        from robotq.core.augmentations.mirror import Mirror

        return Mirror
    elif canon == "ColorJitter":
        from robotq.core.augmentations.color import ColorJitter

        return ColorJitter
    elif canon == "GaussianNoise":
        from robotq.core.augmentations.noise import GaussianNoise

        return GaussianNoise
    elif canon == "ActionNoise":
        from robotq.core.augmentations.noise import ActionNoise

        return ActionNoise
    elif canon == "SpeedWarp":
        from robotq.core.augmentations.speed import SpeedWarp

        return SpeedWarp
    else:
        raise ValueError(f"No import path for {canon!r}")


def _build_pipeline_from_names(
    names: list[str],
    adapter=None,
):
    """Build a Compose pipeline from a list of short augmentation names."""
    from robotq.core.pipeline import Compose
    from robotq.core.transform import RobotTransform

    transforms = []
    for name in names:
        cls = _resolve_augmentation_class(name)
        if issubclass(cls, RobotTransform):
            if adapter is None:
                raise ValueError(
                    f"Augmentation {name!r} requires an adapter, but none was resolved."
                )
            transforms.append(cls(adapter=adapter))
        else:
            transforms.append(cls())
    return Compose(transforms)


# ---------------------------------------------------------------------------
# Tool: list_augmentations
# ---------------------------------------------------------------------------


@mcp.tool()
def list_augmentations() -> list[dict]:
    """List available augmentation transforms with their types and requirements."""
    return [
        {
            "name": "mirror",
            "type": "RobotTransform",
            "adapter": "required",
            "description": "Horizontal flip + L/R arm swap",
        },
        {
            "name": "color_jitter",
            "type": "SequenceTransform",
            "adapter": "none",
            "description": "Brightness/contrast/saturation/hue",
        },
        {
            "name": "gaussian_noise",
            "type": "FrameTransform",
            "adapter": "none",
            "description": "Per-frame pixel noise",
        },
        {
            "name": "action_noise",
            "type": "TrajectoryTransform",
            "adapter": "none",
            "description": "Gaussian perturbation on actions",
        },
        {
            "name": "speed_warp",
            "type": "TrajectoryTransform",
            "adapter": "none",
            "description": "Time-stretch episodes",
        },
    ]


# ---------------------------------------------------------------------------
# Tool: augment_dataset
# ---------------------------------------------------------------------------


@mcp.tool()
def augment_dataset(
    source: str,
    output: str,
    augmentations: list[str],
    adapter: str = "aloha",
    multiply: int = 1,
    max_episodes: Optional[int] = None,
    no_upload: bool = False,
) -> str:
    """Augment a LeRobot v3 dataset and upload to HF Hub.

    Parameters:
        source: HF repo ID, e.g. "lerobot/aloha_static_cups_open"
        output: Output repo ID, e.g. "user/augmented"
        augmentations: List of augmentation names, e.g. ["mirror", "color_jitter"]
        adapter: Robot adapter name (default "aloha")
        multiply: Number of augmented copies per original episode (default 1)
        max_episodes: Maximum number of source episodes to load (None = all)
        no_upload: If True, write locally only, skip push to HF Hub

    Returns:
        URL to the HuggingFace dataset visualizer.
    """
    from robotq.core.config import resolve_adapter
    from robotq.io.loader import load_dataset
    from robotq.io.writer import write_dataset

    # 1. Resolve adapter
    resolved_adapter = resolve_adapter(adapter)

    # 2. Build pipeline from augmentation names
    pipeline = _build_pipeline_from_names(augmentations, adapter=resolved_adapter)

    # 3. Load dataset
    episodes = load_dataset(source, max_episodes=max_episodes)

    # 4. Apply pipeline (multiply times)
    all_episodes = list(episodes)  # originals
    for ep in episodes:
        for _ in range(multiply):
            augmented = pipeline(copy.deepcopy(ep))
            all_episodes.append(augmented)

    # 5. Write dataset
    viz_link = write_dataset(
        all_episodes,
        repo_id=output,
        local_only=no_upload,
    )

    # 6. Return visualizer link
    total = len(all_episodes)
    orig = len(episodes)
    return (
        f"Augmented {orig} episodes x{multiply} -> {total} total episodes. Visualize at: {viz_link}"
    )


# ---------------------------------------------------------------------------
# Tool: preview_augmentation
# ---------------------------------------------------------------------------


@mcp.tool()
def preview_augmentation(
    source: str,
    augmentation: str,
    episode: int = 0,
) -> str:
    """Preview an augmentation on a single episode. Returns a text summary of the changes.

    Parameters:
        source: HF repo ID, e.g. "lerobot/aloha_static_cups_open"
        augmentation: Augmentation name, e.g. "mirror"
        episode: Episode index to preview (default 0)
    """
    import numpy as np

    from robotq.core.config import resolve_adapter
    from robotq.io.loader import load_dataset

    # Load one episode
    episodes = load_dataset(source, max_episodes=episode + 1)
    if episode >= len(episodes):
        return f"Episode {episode} not found. Dataset has {len(episodes)} episodes."

    original = episodes[episode]

    # Build a single-augmentation pipeline
    # Determine if adapter is needed
    from robotq.core.transform import RobotTransform

    cls = _resolve_augmentation_class(augmentation)
    if issubclass(cls, RobotTransform):
        adapter = resolve_adapter("aloha")
        transform = cls(adapter=adapter)
    else:
        transform = cls()

    augmented = transform(copy.deepcopy(original))

    # Compute summary statistics
    lines = [
        f"Augmentation: {augmentation}",
        f"Episode: {episode}",
        f"Original frames: {original.num_frames}",
        f"Augmented frames: {augmented.num_frames}",
        f"Action dim: {original.action_dim}",
        f"State dim: {original.state_dim}",
        f"Cameras: {', '.join(original.metadata.camera_names)}",
    ]

    # Frame-level diff for first camera
    if original.metadata.camera_names:
        cam = original.metadata.camera_names[0]
        orig_first = original.frames[cam][0].astype(np.float32)
        aug_first = augmented.frames[cam][0].astype(np.float32)
        if orig_first.shape == aug_first.shape:
            pixel_diff = np.abs(aug_first - orig_first)
            lines.append(
                f"Pixel diff (first frame, {cam}): "
                f"mean={pixel_diff.mean():.2f}, max={pixel_diff.max():.1f}"
            )
        else:
            lines.append(f"Frame shape changed: {orig_first.shape} -> {aug_first.shape}")

    # Action diff
    if original.num_frames == augmented.num_frames:
        action_diff = np.abs(augmented.actions - original.actions)
        lines.append(
            f"Action diff: mean={action_diff.mean():.6f}, "
            f"max={action_diff.max():.6f}, "
            f"std={action_diff.std():.6f}"
        )
    else:
        lines.append(
            f"Frame count changed ({original.num_frames} -> {augmented.num_frames}), "
            f"action diff not directly comparable."
        )

    # State diff
    if original.num_frames == augmented.num_frames:
        state_diff = np.abs(augmented.states - original.states)
        lines.append(
            f"State diff: mean={state_diff.mean():.6f}, "
            f"max={state_diff.max():.6f}, "
            f"std={state_diff.std():.6f}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point: python -m robotq.mcp.server
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
