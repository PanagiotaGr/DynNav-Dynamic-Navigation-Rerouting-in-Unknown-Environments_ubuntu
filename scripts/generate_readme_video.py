#!/usr/bin/env python3
"""Render the README GIF from a real deterministic DynNav planner execution."""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from dynnav.planners.grid_map import GridCell, GridMap
from dynnav.planners.online_recoverability import (
    ObstacleUpdate,
    OnlineRecoverabilityPlanner,
    OnlineReplanningResult,
)
from dynnav.planners.recoverability_astar import PlannerMode, RecoverabilityAStarConfig
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 960, 540
GIF_SIZE = (640, 360)
GIF_FRAME_STEP = 3
FPS = 8
EVENT_STEP = 5
EVENT_CELL = (10, 5)

COLORS = {
    "background": "#06101c",
    "panel": "#0b1a2a",
    "panel_alt": "#10263a",
    "cell": "#0d2134",
    "grid": "#223f59",
    "wall": "#385068",
    "text": "#f5f9fc",
    "muted": "#91abc0",
    "cyan": "#2ad4c5",
    "blue": "#4b8fff",
    "amber": "#ffbd59",
    "red": "#ff6075",
    "green": "#58d68d",
}


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for directory in (Path("/usr/share/fonts/truetype/dejavu"), Path("/usr/share/fonts/dejavu")):
        candidate = directory / name
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    raise FileNotFoundError(f"Could not find {name}")


FONTS = {
    "title": font(27, bold=True),
    "heading": font(18, bold=True),
    "body": font(14),
    "small": font(11),
    "tiny": font(9),
    "label": font(12, bold=True),
    "metric": font(22, bold=True),
}


@dataclass(frozen=True)
class DemoRun:
    """Traceable inputs and outputs for the README animation."""

    grid: GridMap
    safe_cells: frozenset[GridCell]
    start: GridCell
    goal: GridCell
    event_step: int
    event_cell: GridCell
    result: OnlineReplanningResult


def build_demo_run() -> DemoRun:
    """Execute the canonical J3 planner on a deterministic invalidation case."""
    width, height = 22, 12
    obstacles = (
        {(x, 0) for x in range(width)}
        | {(x, height - 1) for x in range(width)}
        | {(0, y) for y in range(height)}
        | {(width - 1, y) for y in range(height)}
    )
    obstacles |= {(10, y) for y in range(1, height - 1) if y not in {2, 5, 9}}
    obstacles |= {(16, y) for y in range(1, height - 1) if y not in {5, 9}}

    risk = {(x, y): 0.75 for x in range(4, 19) for y in range(1, 4)}
    risk.update({(x, 8): 0.12 for x in range(6, 17)})
    grid = GridMap.from_obstacles(width, height, obstacles, risk=risk)
    start, goal = (1, 5), (20, 5)
    safe_cells = frozenset((1, y) for y in range(3, 8))
    config = RecoverabilityAStarConfig(risk_weight=5.0, irreversibility_weight=2.5)
    result = OnlineRecoverabilityPlanner(
        grid=grid,
        start=start,
        goal=goal,
        mode=PlannerMode.PROPOSED,
        config=config,
        safe_cells=set(safe_cells),
    ).run({EVENT_STEP: (ObstacleUpdate(EVENT_CELL),)}, max_steps=80)

    if not result.success:
        raise RuntimeError(f"README demonstration failed: {result.status.value}")
    if EVENT_CELL not in result.steps[0].selected_path:
        raise RuntimeError("The configured event does not invalidate the initial path")
    event_record = result.steps[EVENT_STEP]
    if not event_record.applied_updates or EVENT_CELL in event_record.selected_path:
        raise RuntimeError("The planner did not produce a repaired path after invalidation")

    return DemoRun(grid, safe_cells, start, goal, EVENT_STEP, EVENT_CELL, result)


def rounded(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: str,
    outline: str | None = None,
    *,
    radius: int = 14,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=2 if outline else 1)


def blend(left: str, right: str, amount: float) -> str:
    """Blend two hexadecimal colours."""
    a = tuple(int(left[index : index + 2], 16) for index in (1, 3, 5))
    b = tuple(int(right[index : index + 2], 16) for index in (1, 3, 5))
    mixed = tuple(round(x + (y - x) * amount) for x, y in zip(a, b, strict=True))
    return "#" + "".join(f"{value:02x}" for value in mixed)


def map_xy(cell: tuple[float, float], grid: GridMap) -> tuple[float, float]:
    x0, y0, cell_size = 52, 127, 24
    x, y = cell
    return x0 + (x + 0.5) * cell_size, y0 + (grid.height - y - 0.5) * cell_size


def draw_path(
    draw: ImageDraw.ImageDraw,
    path: tuple[GridCell, ...] | list[GridCell],
    grid: GridMap,
    *,
    fill: str,
    width: int,
) -> None:
    if len(path) >= 2:
        draw.line([map_xy(cell, grid) for cell in path], fill=fill, width=width, joint="curve")


def draw_dashed_path(
    draw: ImageDraw.ImageDraw,
    path: tuple[GridCell, ...],
    grid: GridMap,
    *,
    fill: str,
) -> None:
    for index, (a, b) in enumerate(zip(path, path[1:], strict=False)):
        if index % 2 == 0:
            draw.line((map_xy(a, grid), map_xy(b, grid)), fill=fill, width=3)


def draw_header(draw: ImageDraw.ImageDraw, phase: str, step: int, total: int) -> None:
    draw.rectangle((0, 0, WIDTH, 8), fill=COLORS["cyan"])
    draw.text((34, 25), "DynNav", font=FONTS["title"], fill=COLORS["text"])
    draw.text((154, 32), "LIVE REPLANNING TRACE", font=FONTS["label"], fill=COLORS["cyan"])
    badges = (
        ("J3 · PROPOSED", COLORS["cyan"]),
        ("DETERMINISTIC", COLORS["blue"]),
        ("SOFTWARE SIM", COLORS["amber"]),
    )
    x = 556
    for label, color in badges:
        box_width = 15 + draw.textlength(label, font=FONTS["tiny"])
        rounded(draw, (round(x), 26, round(x + box_width), 48), COLORS["panel_alt"], color, radius=8)
        draw.text((x + 8, 32), label, font=FONTS["tiny"], fill=color)
        x += box_width + 8
    draw.text((35, 73), phase, font=FONTS["label"], fill=COLORS["text"])
    draw.text((840, 73), f"STEP {step:02d}/{total:02d}", font=FONTS["small"], fill=COLORS["muted"])
    draw.line((35, 96, 925, 96), fill=COLORS["grid"], width=2)


def draw_grid(
    draw: ImageDraw.ImageDraw,
    run: DemoRun,
    *,
    step: int,
    robot: tuple[float, float],
    pulse: float,
) -> None:
    grid = run.grid
    current_obstacles = set(grid.obstacles)
    event_applied = step >= run.event_step
    if event_applied:
        current_obstacles.add(run.event_cell)

    rounded(draw, (34, 108, 600, 440), COLORS["panel"], COLORS["grid"])
    draw.text((52, 115), "OCCUPANCY + NORMALIZED RISK", font=FONTS["tiny"], fill=COLORS["muted"])
    for x in range(grid.width):
        for y in range(grid.height):
            left = 52 + x * 24
            top = 127 + (grid.height - y - 1) * 24
            cell = (x, y)
            if cell in current_obstacles:
                color = COLORS["wall"]
            else:
                risk = grid.cell_risk(cell)
                color = blend(COLORS["cell"], COLORS["amber"], min(0.66, risk * 0.75))
                if cell in run.safe_cells:
                    color = blend(color, COLORS["cyan"], 0.28)
            if event_applied and cell == run.event_cell:
                color = COLORS["red"]
            draw.rectangle((left, top, left + 24, top + 24), fill=color, outline=COLORS["grid"])

    initial_path = run.result.steps[0].selected_path
    record = run.result.steps[min(step, len(run.result.steps) - 1)]
    if event_applied:
        draw_dashed_path(draw, initial_path, grid, fill=COLORS["amber"])
    draw_path(draw, record.selected_path, grid, fill=COLORS["cyan"], width=5)
    executed = list(run.result.trajectory[: min(step + 1, len(run.result.trajectory))])
    if executed:
        draw_path(draw, executed, grid, fill=COLORS["blue"], width=6)

    sx, sy = map_xy(run.start, grid)
    gx, gy = map_xy(run.goal, grid)
    draw.ellipse((sx - 8, sy - 8, sx + 8, sy + 8), fill=COLORS["green"], outline=COLORS["text"], width=2)
    draw.text((sx - 4, sy - 5), "S", font=FONTS["tiny"], fill=COLORS["background"])
    draw.ellipse((gx - 9, gy - 9, gx + 9, gy + 9), fill=COLORS["amber"], outline=COLORS["text"], width=2)
    draw.text((gx - 4, gy - 5), "G", font=FONTS["tiny"], fill=COLORS["background"])

    if event_applied:
        ex, ey = map_xy(run.event_cell, grid)
        radius = 13 + round(3 * pulse)
        draw.ellipse((ex - radius, ey - radius, ex + radius, ey + radius), outline=COLORS["red"], width=3)

    rx, ry = map_xy(robot, grid)
    draw.ellipse((rx - 12, ry - 12, rx + 12, ry + 12), fill=COLORS["blue"], outline=COLORS["text"], width=3)
    draw.ellipse((rx - 3, ry - 3, rx + 3, ry + 3), fill=COLORS["text"])

    draw.line((55, 425, 78, 425), fill=COLORS["cyan"], width=5)
    draw.text((84, 418), "selected route", font=FONTS["tiny"], fill=COLORS["muted"])
    draw.line((195, 425, 218, 425), fill=COLORS["blue"], width=5)
    draw.text((224, 418), "executed", font=FONTS["tiny"], fill=COLORS["muted"])
    draw.rectangle((314, 417, 326, 429), fill=COLORS["amber"])
    draw.text((333, 418), "risk", font=FONTS["tiny"], fill=COLORS["muted"])
    draw.rectangle((390, 417, 402, 429), fill=COLORS["red"])
    draw.text((409, 418), "new obstacle", font=FONTS["tiny"], fill=COLORS["muted"])


def metric(draw: ImageDraw.ImageDraw, x: int, y: int, value: str, label: str, color: str) -> None:
    draw.text((x, y), value, font=FONTS["metric"], fill=color)
    draw.text((x, y + 28), label, font=FONTS["tiny"], fill=COLORS["muted"])


def draw_metrics(draw: ImageDraw.ImageDraw, run: DemoRun, *, step: int, phase: str) -> None:
    record = run.result.steps[min(step, len(run.result.steps) - 1)]
    event_applied = step >= run.event_step
    rounded(draw, (622, 108, 926, 440), COLORS["panel"], COLORS["grid"])
    draw.text((646, 126), "PLANNER TELEMETRY", font=FONTS["tiny"], fill=COLORS["muted"])
    draw.text((646, 151), "Risk + recoverability A*", font=FONTS["heading"], fill=COLORS["text"])
    draw.text((646, 178), "g + λᵣ·risk + λᵢ·irreversibility", font=FONTS["small"], fill=COLORS["cyan"])

    metric(draw, 646, 215, str(max(0, len(record.selected_path) - 1)), "remaining path cells", COLORS["cyan"])
    metric(draw, 790, 215, str(record.nodes_expanded), "nodes expanded", COLORS["blue"])
    metric(draw, 646, 276, f"{record.cumulative_risk:.2f}", "planned risk", COLORS["amber"])
    metric(draw, 790, 276, str(record.escape_options), "local escape options", COLORS["green"])

    draw.line((646, 337, 902, 337), fill=COLORS["grid"], width=2)
    status_color = (
        COLORS["red"]
        if step == run.event_step
        else COLORS["green"]
        if phase == "GOAL REACHED"
        else COLORS["cyan"]
    )
    draw.ellipse((647, 355, 661, 369), fill=status_color)
    draw.text((674, 353), "MAP UPDATE", font=FONTS["tiny"], fill=COLORS["muted"])
    draw.text(
        (674, 373),
        "1 obstacle inserted" if event_applied else "awaiting route event",
        font=FONTS["body"],
        fill=COLORS["text"],
    )
    draw.text((646, 407), "Initial route blocked → lower-risk repair", font=FONTS["small"], fill=COLORS["amber"])


def render_frame(
    run: DemoRun,
    *,
    step: int,
    robot: tuple[float, float],
    phase: str,
    pulse: float = 0.0,
) -> Image.Image:
    """Render one frame from a real planner record."""
    image = Image.new("RGB", (WIDTH, HEIGHT), COLORS["background"])
    draw = ImageDraw.Draw(image)
    draw_header(draw, phase, step, run.result.path_length)
    draw_grid(draw, run, step=step, robot=robot, pulse=pulse)
    draw_metrics(draw, run, step=step, phase=phase)
    draw.rectangle((0, 475, WIDTH, HEIGHT), fill="#040a12")
    draw.text(
        (34, 489),
        "CANONICAL PYTHON PLANNER OUTPUT  ·  DETERMINISTIC SOFTWARE SIMULATION",
        font=FONTS["small"],
        fill=COLORS["text"],
    )
    draw.text(
        (34, 513),
        "Generated from OnlineRecoverabilityPlanner records · not Gazebo or physical-robot footage",
        font=FONTS["tiny"],
        fill=COLORS["muted"],
    )
    return image


def interpolate(a: GridCell, b: GridCell, amount: float) -> tuple[float, float]:
    return a[0] + (b[0] - a[0]) * amount, a[1] + (b[1] - a[1]) * amount


def render_frames(run: DemoRun | None = None) -> list[Image.Image]:
    """Turn the planner trajectory and per-step routes into animation frames."""
    run = run or build_demo_run()
    frames: list[Image.Image] = []
    start = run.result.trajectory[0]
    for frame in range(7):
        frames.append(render_frame(run, step=0, robot=start, phase="INITIAL PLAN COMPUTED", pulse=frame / 6))

    for step, record in enumerate(run.result.steps):
        current = run.result.trajectory[step]
        following = run.result.trajectory[step + 1]
        if record.applied_updates:
            for frame in range(7):
                frames.append(
                    render_frame(
                        run,
                        step=step,
                        robot=current,
                        phase="ROUTE INVALIDATED → REPLANNING",
                        pulse=0.5 + 0.5 * math.sin(frame / 7 * 2 * math.pi),
                    )
                )
        phase = "EXECUTING INITIAL ROUTE" if step < run.event_step else "EXECUTING REPAIRED ROUTE"
        for subframe in range(2):
            frames.append(
                render_frame(
                    run,
                    step=step,
                    robot=interpolate(current, following, (subframe + 1) / 2),
                    phase=phase,
                    pulse=subframe / 2,
                )
            )

    goal = run.result.trajectory[-1]
    for frame in range(10):
        frames.append(
            render_frame(
                run,
                step=len(run.result.steps) - 1,
                robot=goal,
                phase="GOAL REACHED",
                pulse=frame / 9,
            )
        )
    return frames


def write_gif(frames: list[Image.Image], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    quantized = [
        frame.resize(GIF_SIZE, Image.Resampling.LANCZOS).quantize(
            colors=32,
            method=Image.Quantize.MEDIANCUT,
            dither=Image.Dither.NONE,
        )
        for frame in frames[::GIF_FRAME_STEP]
    ]
    quantized[0].save(
        destination,
        save_all=True,
        append_images=quantized[1:],
        duration=round(1000 * GIF_FRAME_STEP / FPS),
        loop=0,
        optimize=True,
        disposal=2,
    )


def write_mp4(frames: list[Image.Image], destination: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to generate the MP4")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dynnav-planner-video-") as directory:
        frame_dir = Path(directory)
        for index, frame in enumerate(frames):
            frame.save(frame_dir / f"frame_{index:03d}.png", optimize=True)
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                str(FPS),
                "-i",
                str(frame_dir / "frame_%03d.png"),
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-metadata",
                "comment=Canonical DynNav J3 deterministic software simulation; not Gazebo or hardware footage",
                str(destination),
            ],
            check=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gif", type=Path, default=ROOT / "assets/dynnav_system_overview.gif")
    parser.add_argument("--mp4", type=Path, default=ROOT / "assets/dynnav_system_overview.mp4")
    args = parser.parse_args()

    run = build_demo_run()
    frames = render_frames(run)
    write_gif(frames, args.gif)
    write_mp4(frames, args.mp4)
    print(
        f"Generated {len(frames)} frames from {len(run.result.steps)} planner records: "
        f"{args.gif} and {args.mp4}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
