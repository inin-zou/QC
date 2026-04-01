"""RobotQ CLI — Typer application for dataset augmentation.

Entry point registered in pyproject.toml:
    robotq = "robotq.cli.main:app"
"""

from __future__ import annotations

import copy
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

app = typer.Typer(
    name="robotq",
    help="Composable augmentation toolkit for LeRobot v3 datasets.",
    add_completion=False,
)
console = Console()


# ---------------------------------------------------------------------------
# robotq augment
# ---------------------------------------------------------------------------

@app.command()
def augment(
    dataset: str = typer.Option(..., help="HF repo ID, e.g. lerobot/aloha_static_cups_open"),
    output: str = typer.Option(..., help="Output HF repo ID, e.g. user/augmented-dataset"),
    mirror: bool = typer.Option(False, "--mirror", help="Enable Mirror augmentation"),
    color_jitter: bool = typer.Option(False, "--color-jitter", help="Enable ColorJitter"),
    gaussian_noise: bool = typer.Option(False, "--gaussian-noise", help="Enable GaussianNoise"),
    action_noise: bool = typer.Option(False, "--action-noise", help="Enable ActionNoise"),
    speed_warp: bool = typer.Option(False, "--speed-warp", help="Enable SpeedWarp (0.8-1.2x)"),
    config: Optional[str] = typer.Option(None, help="Path to YAML config file (overrides flags)"),
    adapter: str = typer.Option("aloha", help="Robot adapter name"),
    multiply: int = typer.Option(1, help="Augmented copies per original episode"),
    max_episodes: Optional[int] = typer.Option(None, help="Limit number of source episodes"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print summary, don't process"),
    no_upload: bool = typer.Option(False, "--no-upload", help="Write locally, skip push_to_hub"),
    token: Optional[str] = typer.Option(None, help="HF token for upload"),
) -> None:
    """Augment a LeRobot v3 dataset with composable transforms."""
    from robotq.core.pipeline import Compose

    # -- 1. Read config first (may override dataset/output/adapter/multiply) -----
    pipeline: Compose | None = None

    if config is not None:
        from robotq.core.config import load_config, build_pipeline, resolve_adapter

        cfg = load_config(config)
        dataset = cfg.get("dataset", dataset)
        output = cfg.get("output", output)
        adapter = cfg.get("adapter", adapter)
        multiply = cfg.get("multiply", multiply)
        resolved_adapter = resolve_adapter(adapter)
        pipeline = build_pipeline(cfg, adapter=resolved_adapter)
        console.print(f"[bold blue]Pipeline loaded from config:[/] {config}")

    # -- 2. Load dataset ---------------------------------------------------------
    console.print(f"[bold blue]Loading dataset:[/] {dataset}")
    from robotq.io.loader import load_dataset as _load_dataset

    episodes = _load_dataset(dataset, max_episodes=max_episodes)
    console.print(
        f"  Loaded [green]{len(episodes)}[/] episode(s), "
        f"[green]{episodes[0].num_frames}[/] frames in first episode"
    )

    # -- 3. Build pipeline from flags (if no config) ----------------------------
    if pipeline is None:
        # Build pipeline from CLI flags
        resolved_adapter = _resolve_adapter(adapter)
        transforms = _build_transforms_from_flags(
            mirror=mirror,
            color_jitter=color_jitter,
            gaussian_noise=gaussian_noise,
            action_noise=action_noise,
            speed_warp=speed_warp,
            adapter=resolved_adapter,
        )
        if transforms:
            pipeline = Compose(transforms)
            console.print(
                f"[bold blue]Pipeline:[/] Compose({len(transforms)} transform(s))"
            )
        else:
            console.print("[yellow]No augmentations selected — output will be a copy.[/]")

    # -- 4. Dry-run summary ------------------------------------------------------
    if dry_run:
        _print_dry_run_summary(
            dataset=dataset,
            output=output,
            episodes=episodes,
            pipeline=pipeline,
            multiply=multiply,
            no_upload=no_upload,
        )
        raise typer.Exit()

    # -- 5. Apply pipeline -------------------------------------------------------
    all_episodes = list(episodes)  # start with originals

    if pipeline is not None:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Augmenting episodes", total=len(episodes) * multiply)
            for ep in episodes:
                for _ in range(multiply):
                    augmented = pipeline(copy.deepcopy(ep))
                    all_episodes.append(augmented)
                    progress.advance(task)

    console.print(
        f"[bold green]Total episodes:[/] {len(all_episodes)} "
        f"({len(episodes)} original + {len(all_episodes) - len(episodes)} augmented)"
    )

    # -- 6. Write dataset --------------------------------------------------------
    console.print(f"[bold blue]Writing dataset to:[/] {output}")
    from robotq.io.writer import write_dataset as _write_dataset, generate_visualizer_link

    viz_link = _write_dataset(
        all_episodes,
        repo_id=output,
        local_only=no_upload,
        token=token,
    )

    # -- 7. Print visualizer link ------------------------------------------------
    console.print(f"[bold green]Done![/] Visualize at: {viz_link}")


# ---------------------------------------------------------------------------
# robotq preview
# ---------------------------------------------------------------------------

@app.command()
def preview(
    dataset: str = typer.Option(..., help="HF repo ID"),
    episode: int = typer.Option(0, help="Episode index to preview"),
    mirror: bool = typer.Option(False, "--mirror", help="Enable Mirror"),
    color_jitter: bool = typer.Option(False, "--color-jitter", help="Enable ColorJitter"),
    gaussian_noise: bool = typer.Option(False, "--gaussian-noise", help="Enable GaussianNoise"),
    action_noise: bool = typer.Option(False, "--action-noise", help="Enable ActionNoise"),
    speed_warp: bool = typer.Option(False, "--speed-warp", help="Enable SpeedWarp"),
    config: Optional[str] = typer.Option(None, help="Path to YAML config"),
    adapter: str = typer.Option("aloha", help="Robot adapter name"),
) -> None:
    """Preview augmentation on a single episode. Saves before/after PNGs."""
    import copy
    import os
    import numpy as np
    import cv2
    from robotq.core.pipeline import Compose

    # -- 1. Load dataset (just enough episodes to reach the requested index) ----
    console.print(f"[bold blue]Loading dataset:[/] {dataset}  (episode {episode})")
    from robotq.io.loader import load_dataset as _load_dataset

    episodes = _load_dataset(dataset, max_episodes=episode + 1)
    original = episodes[episode]
    console.print(
        f"  Loaded episode {episode}: [green]{original.num_frames}[/] frames, "
        f"action_dim={original.action_dim}, state_dim={original.state_dim}"
    )

    # -- 2. Build pipeline -------------------------------------------------------
    pipeline: Compose | None = None

    if config is not None:
        from robotq.core.config import load_config, build_pipeline, resolve_adapter

        cfg = load_config(config)
        adapter = cfg.get("adapter", adapter)
        resolved_adapter = resolve_adapter(adapter)
        pipeline = build_pipeline(cfg, adapter=resolved_adapter)
        console.print(f"[bold blue]Pipeline loaded from config:[/] {config}")
    else:
        resolved_adapter = _resolve_adapter(adapter)
        transforms = _build_transforms_from_flags(
            mirror=mirror,
            color_jitter=color_jitter,
            gaussian_noise=gaussian_noise,
            action_noise=action_noise,
            speed_warp=speed_warp,
            adapter=resolved_adapter,
        )
        if transforms:
            pipeline = Compose(transforms)
            console.print(
                f"[bold blue]Pipeline:[/] Compose({len(transforms)} transform(s))"
            )
        else:
            console.print("[yellow]No augmentations selected — augmented output will equal original.[/]")

    # -- 3. Apply pipeline -------------------------------------------------------
    augmented = pipeline(copy.deepcopy(original)) if pipeline is not None else copy.deepcopy(original)

    # -- 4. Create preview/ directory -------------------------------------------
    os.makedirs("preview", exist_ok=True)

    # -- 5. Save before/after frames for the first camera -----------------------
    camera_names = original.metadata.camera_names
    if not camera_names:
        console.print("[yellow]No cameras found in episode metadata — skipping frame export.[/]")
    else:
        cam = camera_names[0]
        orig_frames = original.frames[cam]   # list/array of (H, W, 3) uint8 RGB
        aug_frames = augmented.frames[cam]

        n_orig = len(orig_frames)
        n_aug = len(aug_frames)

        # Sample indices: first, middle, last (clamped to actual length)
        sample_labels = [0, 200, 399]
        saved_files: list[str] = []

        for label in sample_labels:
            # Before frame
            orig_idx = min(label, n_orig - 1)
            before_frame = np.array(orig_frames[orig_idx])
            before_bgr = cv2.cvtColor(before_frame, cv2.COLOR_RGB2BGR)
            before_path = f"preview/before_frame_{label:03d}.png"
            cv2.imwrite(before_path, before_bgr)
            saved_files.append(before_path)

            # After frame
            aug_idx = min(label, n_aug - 1)
            after_frame = np.array(aug_frames[aug_idx])
            after_bgr = cv2.cvtColor(after_frame, cv2.COLOR_RGB2BGR)
            after_path = f"preview/after_frame_{label:03d}.png"
            cv2.imwrite(after_path, after_bgr)
            saved_files.append(after_path)

        console.print(f"[bold blue]Camera:[/] {cam!r} — saved {len(saved_files)} PNGs to preview/")

    # -- 6. Print summary --------------------------------------------------------
    orig_action = np.array(original.action)     # (T, D)
    aug_action = np.array(augmented.action)

    # Align lengths for diff (SpeedWarp may change frame count)
    min_t = min(len(orig_action), len(aug_action))
    action_diff = float(np.mean(np.abs(orig_action[:min_t] - aug_action[:min_t])))

    orig_state = np.array(original.state)
    aug_state = np.array(augmented.state)
    min_ts = min(len(orig_state), len(aug_state))
    state_diff = float(np.mean(np.abs(orig_state[:min_ts] - aug_state[:min_ts])))

    from rich.table import Table as _Table

    summary = _Table(title="Preview Summary", show_lines=True)
    summary.add_column("Property", style="bold")
    summary.add_column("Value")

    summary.add_row("Dataset", dataset)
    summary.add_row("Episode index", str(episode))
    summary.add_row("Frame count (original)", str(original.num_frames))
    summary.add_row("Frame count (augmented)", str(augmented.num_frames))
    summary.add_row("Action diff (mean |Δ|)", f"{action_diff:.6f}")
    summary.add_row("State diff (mean |Δ|)", f"{state_diff:.6f}")
    if camera_names:
        summary.add_row("Saved files", "\n".join(saved_files))

    console.print(summary)


# ---------------------------------------------------------------------------
# robotq list
# ---------------------------------------------------------------------------

@app.command(name="list")
def list_augmentations() -> None:
    """Show available augmentations."""
    table = Table(title="Available Augmentations")
    table.add_column("Name", style="bold cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Adapter", style="green")
    table.add_column("Dependencies")

    table.add_row("Mirror", "RobotTransform", "required", "built-in")
    table.add_row("ColorJitter", "SequenceTransform", "-", "built-in")
    table.add_row("GaussianNoise", "FrameTransform", "-", "built-in")
    table.add_row("ActionNoise", "TrajectoryTransform", "-", "built-in")
    table.add_row("SpeedWarp", "TrajectoryTransform", "-", "built-in")

    console.print(table)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_adapter(name: str):
    """Resolve an adapter name to an adapter instance."""
    if name == "aloha":
        from robotq.adapters.aloha import AlohaAdapter

        return AlohaAdapter()
    raise typer.BadParameter(f"Unknown adapter: {name!r}. Available: aloha")


def _build_transforms_from_flags(
    *,
    mirror: bool,
    color_jitter: bool,
    gaussian_noise: bool,
    action_noise: bool,
    speed_warp: bool,
    adapter,
) -> list:
    """Instantiate transforms based on CLI boolean flags."""
    transforms = []

    if mirror:
        from robotq.core.augmentations.mirror import Mirror

        transforms.append(Mirror(adapter=adapter))

    if color_jitter:
        from robotq.core.augmentations.color import ColorJitter

        transforms.append(ColorJitter())

    if gaussian_noise:
        from robotq.core.augmentations.noise import GaussianNoise

        transforms.append(GaussianNoise())

    if action_noise:
        from robotq.core.augmentations.noise import ActionNoise

        transforms.append(ActionNoise())

    if speed_warp:
        from robotq.core.augmentations.speed import SpeedWarp

        transforms.append(SpeedWarp())

    return transforms


def _print_dry_run_summary(
    *,
    dataset: str,
    output: str,
    episodes: list,
    pipeline,
    multiply: int,
    no_upload: bool,
) -> None:
    """Print a Rich summary table for --dry-run and exit."""
    from robotq.core.pipeline import Compose

    table = Table(title="Dry Run Summary", show_lines=True)
    table.add_column("Property", style="bold")
    table.add_column("Value")

    table.add_row("Source dataset", dataset)
    table.add_row("Output dataset", output)
    table.add_row("Source episodes", str(len(episodes)))

    first = episodes[0]
    table.add_row("Frames per episode (first)", str(first.num_frames))
    table.add_row("Action dim", str(first.action_dim))
    table.add_row("State dim", str(first.state_dim))
    table.add_row("Camera names", ", ".join(first.metadata.camera_names))
    table.add_row("Robot type", first.metadata.robot_type)
    table.add_row("FPS", str(first.metadata.fps))

    if pipeline is not None and isinstance(pipeline, Compose):
        transform_names = [repr(t) for t in pipeline.transforms]
        table.add_row("Pipeline transforms", "\n".join(transform_names))
    elif pipeline is not None:
        table.add_row("Pipeline", repr(pipeline))
    else:
        table.add_row("Pipeline transforms", "(none)")

    table.add_row("Multiply", str(multiply))

    total = len(episodes) + len(episodes) * multiply
    if pipeline is None:
        total = len(episodes)
    table.add_row("Expected output episodes", str(total))
    table.add_row("Upload", "no" if no_upload else "yes")

    console.print(table)
