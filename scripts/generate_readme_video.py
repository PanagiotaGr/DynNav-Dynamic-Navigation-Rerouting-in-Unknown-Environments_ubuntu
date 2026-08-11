#!/usr/bin/env python3
"""Generate the README's deterministic technical overview GIF and MP4."""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 960, 540
GIF_SIZE = (640, 360)
GIF_FRAME_STEP = 2
FPS = 5
FRAMES_PER_SCENE = 10
TOTAL_SCENES = 6
COLORS = {
    "background": "#07111f",
    "panel": "#0d2037",
    "panel_alt": "#102944",
    "text": "#f3f7fb",
    "muted": "#9fb5c9",
    "cyan": "#29d3c2",
    "blue": "#4c8dff",
    "amber": "#ffbd59",
    "red": "#ff6978",
    "grid": "#24415e",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu") / name,
        Path("/usr/share/fonts/dejavu") / name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    raise FileNotFoundError(f"Could not find {name}")


FONTS = {
    "title": font(38, bold=True),
    "heading": font(25, bold=True),
    "body": font(17),
    "small": font(13),
    "label": font(14, bold=True),
}


def rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str, outline: str | None = None) -> None:
    draw.rounded_rectangle(box, radius=16, fill=fill, outline=outline, width=2 if outline else 1)


def base_frame(scene: int, progress: float) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), COLORS["background"])
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 7), fill=COLORS["cyan"])
    draw.text((36, 25), "DynNav", font=FONTS["heading"], fill=COLORS["text"])
    draw.text((850, 31), f"{scene + 1:02d}/{TOTAL_SCENES:02d}", font=FONTS["small"], fill=COLORS["muted"])
    bar_width = int((scene + progress) / TOTAL_SCENES * (WIDTH - 72))
    draw.rounded_rectangle((36, 66, WIDTH - 36, 70), radius=2, fill=COLORS["grid"])
    draw.rounded_rectangle((36, 66, 36 + bar_width, 70), radius=2, fill=COLORS["cyan"])
    draw.rectangle((0, HEIGHT - 34, WIDTH, HEIGHT), fill="#050b14")
    draw.text(
        (24, HEIGHT - 25),
        "TECHNICAL OVERVIEW  |  SYNTHETIC VIGNETTE  |  NOT RECORDED GAZEBO OR REAL-ROBOT EVIDENCE",
        font=FONTS["small"],
        fill=COLORS["muted"],
    )
    return image, draw


def scene_landing(draw: ImageDraw.ImageDraw, progress: float) -> None:
    """Render a deterministic representation of the deployed research landing page."""
    rounded(draw, (30, 90, 930, 490), "#080f18", COLORS["grid"])
    draw.rectangle((30, 90, 930, 132), fill="#0b141f")
    draw.ellipse((48, 105, 62, 119), fill=COLORS["cyan"])
    draw.text((72, 101), "DynNav", font=FONTS["label"], fill=COLORS["text"])
    draw.text((405, 103), "ARCHITECTURE     EVIDENCE     INTERFACES", font=FONTS["small"], fill=COLORS["muted"])
    rounded(draw, (782, 99, 910, 124), COLORS["panel_alt"], COLORS["cyan"])
    draw.text((798, 105), "OPEN RESEARCHER", font=font(9, bold=True), fill=COLORS["cyan"])

    draw.text((58, 165), "RECOVERABILITY-AWARE AUTONOMOUS NAVIGATION", font=font(10, bold=True), fill=COLORS["cyan"])
    draw.text((58, 197), "Plan toward the goal.", font=font(30, bold=True), fill=COLORS["text"])
    draw.text((58, 235), "Preserve a way back.", font=font(30, bold=True), fill=COLORS["cyan"])
    draw.text((58, 291), "ROS 2 + Python research programme for risk,", font=FONTS["small"], fill=COLORS["muted"])
    draw.text((58, 313), "recoverability and online replanning.", font=FONTS["small"], fill=COLORS["muted"])
    rounded(draw, (58, 348, 218, 384), COLORS["cyan"])
    draw.text((79, 359), "EXPLORE RESEARCHER", font=font(10, bold=True), fill=COLORS["background"])
    draw.text(
        (58, 426),
        "26 contributions     4 objectives     ROS 2 Jazzy verified",
        font=FONTS["small"],
        fill=COLORS["muted"],
    )

    rounded(draw, (565, 158, 898, 433), COLORS["panel"], COLORS["grid"])
    draw.text((584, 175), "Synthetic route vignette", font=font(10, bold=True), fill=COLORS["muted"])
    for row in range(5):
        for col in range(6):
            x, y = 584 + col * 48, 207 + row * 40
            draw.rectangle((x, y, x + 48, y + 40), outline=COLORS["grid"])
    risky = [(600, 375), (600, 294), (742, 294), (742, 240), (860, 240), (860, 213)]
    safe = [(600, 375), (742, 375), (742, 345), (860, 345), (860, 213)]
    draw.line(risky, fill=COLORS["amber"], width=4)
    draw.line(safe, fill=COLORS["cyan"], width=6)
    robot_index = min(int(progress * (len(safe) - 1)), len(safe) - 2)
    fraction = progress * (len(safe) - 1) - robot_index
    ax, ay = safe[robot_index]
    bx, by = safe[robot_index + 1]
    rx = int(ax + (bx - ax) * fraction)
    ry = int(ay + (by - ay) * fraction)
    draw.ellipse((rx - 9, ry - 9, rx + 9, ry + 9), fill=COLORS["blue"], outline="white", width=2)


def scene_researcher(draw: ImageDraw.ImageDraw, progress: float) -> None:
    """Render the real Researcher workspace structure in the README walkthrough."""
    rounded(draw, (24, 88, 936, 490), "#080c12", COLORS["grid"])
    draw.rectangle((24, 88, 936, 130), fill="#0d141d")
    draw.text((45, 101), "DynNav Researcher", font=FONTS["label"], fill=COLORS["text"])
    draw.text((362, 103), "researcher  |  commit  |  Python  |  idle", font=font(10), fill=COLORS["muted"])
    rounded(draw, (794, 98, 914, 122), COLORS["panel_alt"], COLORS["grid"])
    draw.text((814, 104), "EXPORT REPORT", font=font(9, bold=True), fill=COLORS["muted"])
    draw.rectangle((24, 130, 936, 158), fill="#0a1017")
    draw.text(
        (45, 139),
        "RESEARCH WORKSPACE     EXPERIMENTS     SCENARIOS     RESULTS     REPRODUCIBILITY",
        font=font(9, bold=True),
        fill=COLORS["muted"],
    )

    draw.rectangle((24, 158, 188, 490), fill="#0d141d", outline=COLORS["grid"])
    rounded(draw, (39, 176, 173, 205), COLORS["panel_alt"], COLORS["cyan"])
    draw.text((62, 185), "+  NEW SESSION", font=font(9, bold=True), fill=COLORS["cyan"])
    draw.text((40, 229), "SAVED SESSIONS", font=font(9, bold=True), fill=COLORS["muted"])
    rounded(draw, (39, 246, 173, 292), COLORS["panel_alt"], COLORS["grid"])
    draw.text((51, 256), "Four-planner study", font=font(10, bold=True), fill=COLORS["text"])
    draw.text((51, 275), "Configured · unexecuted", font=font(8), fill=COLORS["muted"])
    draw.text((40, 322), "RESEARCH ASSETS", font=font(9, bold=True), fill=COLORS["muted"])
    for index, label in enumerate(("Recent experiments", "Scenario library", "Planner presets", "Report history")):
        draw.text((49, 349 + index * 26), label, font=font(9), fill=COLORS["muted"])

    draw.rectangle((188, 158, 714, 490), fill="#090e14", outline=COLORS["grid"])
    draw.text(
        (220, 178),
        "01 DEFINE    02 DESIGN    03 EXECUTE    04 ANALYSE    05 REPORT",
        font=font(9, bold=True),
        fill=COLORS["muted"],
    )
    draw.text((228, 237), "EVIDENCE-BOUND PROTOCOL COMPILER", font=font(10, bold=True), fill=COLORS["cyan"])
    draw.text((228, 273), "What should DynNav investigate?", font=font(25, bold=True), fill=COLORS["text"])
    draw.text((228, 313), "Describe a comparison. The researcher makes the", font=font(11), fill=COLORS["muted"])
    draw.text((228, 333), "protocol explicit before any simulation executes.", font=font(11), fill=COLORS["muted"])
    prompts = (
        "Compare all four planners",
        "Test irreversibility weight",
        "Analyse escape options",
        "Generate benchmark report",
    )
    for index, label in enumerate(prompts):
        x = 228 + (index % 2) * 220
        y = 367 + (index // 2) * 51
        active = index <= int(progress * len(prompts))
        rounded(draw, (x, y, x + 205, y + 40), COLORS["panel_alt"], COLORS["cyan"] if active else COLORS["grid"])
        draw.text((x + 12, y + 13), label, font=font(9), fill=COLORS["text"])

    draw.rectangle((714, 158, 936, 490), fill="#0d141d", outline=COLORS["grid"])
    draw.text((735, 180), "EXPERIMENT INSPECTOR", font=font(9, bold=True), fill=COLORS["muted"])
    draw.text((735, 226), "Evidence boundary", font=FONTS["label"], fill=COLORS["amber"])
    draw.text((735, 257), "Synthetic software", font=font(11), fill=COLORS["muted"])
    draw.text((735, 276), "experiments only.", font=font(11), fill=COLORS["muted"])
    draw.text((735, 313), "4 planners", font=font(11), fill=COLORS["cyan"])
    draw.text((735, 337), "paired seeds", font=font(11), fill=COLORS["cyan"])
    draw.text((735, 361), "provenance bundle", font=font(11), fill=COLORS["cyan"])
    rounded(draw, (735, 428, 913, 466), COLORS["cyan"])
    draw.text((782, 440), "RUN EXPERIMENT", font=font(10, bold=True), fill=COLORS["background"])


def draw_route_vignette(draw: ImageDraw.ImageDraw, progress: float) -> None:
    x0, y0, cell = 520, 106, 39
    rounded(draw, (500, 88, 925, 485), COLORS["panel"], COLORS["grid"])
    draw.text((520, 104), "Synthetic route vignette", font=FONTS["label"], fill=COLORS["muted"])
    for row in range(8):
        for col in range(9):
            x, y = x0 + col * cell, y0 + 30 + row * cell
            draw.rectangle((x, y, x + cell, y + cell), outline=COLORS["grid"], width=1)
    obstacles = {(1, 1), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (6, 3), (6, 4), (2, 5), (3, 5), (4, 5)}
    for row, col in obstacles:
        x, y = x0 + col * cell, y0 + 30 + row * cell
        draw.rectangle((x + 2, y + 2, x + cell - 2, y + cell - 2), fill="#36516d")
    risk_cell = (4, 4)
    rx, ry = x0 + risk_cell[1] * cell + cell // 2, y0 + 30 + risk_cell[0] * cell + cell // 2
    pulse = 10 + int(5 * (1 + math.sin(progress * 2 * math.pi)))
    draw.ellipse((rx - pulse, ry - pulse, rx + pulse, ry + pulse), fill=COLORS["red"])

    safe_path = [(0, 7), (0, 6), (0, 5), (1, 5), (2, 5), (3, 5), (3, 6), (3, 7), (3, 8), (4, 8), (5, 8), (6, 8), (7, 8)]
    risky_path = [
        (0, 7),
        (1, 7),
        (2, 7),
        (3, 7),
        (4, 7),
        (4, 6),
        (4, 5),
        (4, 4),
        (4, 3),
        (5, 3),
        (6, 3),
        (7, 3),
        (7, 4),
        (7, 5),
        (7, 6),
        (7, 7),
        (7, 8),
    ]

    def xy(point: tuple[int, int]) -> tuple[int, int]:
        row, col = point
        return x0 + col * cell + cell // 2, y0 + 30 + row * cell + cell // 2

    draw.line([xy(p) for p in risky_path], fill=COLORS["amber"], width=4)
    draw.line([xy(p) for p in safe_path], fill=COLORS["cyan"], width=6)
    position = min(int(progress * (len(safe_path) - 1)), len(safe_path) - 2)
    fraction = progress * (len(safe_path) - 1) - position
    ax, ay = xy(safe_path[position])
    bx, by = xy(safe_path[position + 1])
    robot = (int(ax + (bx - ax) * fraction), int(ay + (by - ay) * fraction))
    draw.ellipse(
        (robot[0] - 10, robot[1] - 10, robot[0] + 10, robot[1] + 10), fill=COLORS["blue"], outline="white", width=2
    )


def scene_research(draw: ImageDraw.ImageDraw, progress: float) -> None:
    draw.text((38, 105), "Planning that keeps", font=FONTS["title"], fill=COLORS["text"])
    draw.text((38, 150), "escape options open", font=FONTS["title"], fill=COLORS["cyan"])
    draw.text((40, 216), "Shortest is not always safest.", font=FONTS["body"], fill=COLORS["text"])
    draw.text((40, 250), "DynNav scores path length, occupancy risk,", font=FONTS["body"], fill=COLORS["muted"])
    draw.text((40, 277), "and recoverability under route invalidation.", font=FONTS["body"], fill=COLORS["muted"])
    for index, (label, color) in enumerate(
        (("risk", COLORS["amber"]), ("recoverability", COLORS["cyan"]), ("replanning", COLORS["blue"]))
    ):
        y = 338 + index * 42
        draw.ellipse((43, y + 4, 57, y + 18), fill=color)
        draw.text((70, y), label, font=FONTS["body"], fill=COLORS["text"])
    draw_route_vignette(draw, progress)


def scene_pipeline(draw: ImageDraw.ImageDraw, progress: float) -> None:
    draw.text((38, 102), "Executable planning pipeline", font=FONTS["title"], fill=COLORS["text"])
    stages = [
        ("OBSERVE", "Occupancy + dynamic events"),
        ("ESTIMATE", "Risk + recoverability"),
        ("PLAN", "Shortest / risk / recovery / joint"),
        ("EXECUTE", "Nav2 NavigateToPose + evidence"),
    ]
    for index, (name, detail) in enumerate(stages):
        x = 42 + (index % 2) * 450
        y = 178 + (index // 2) * 132
        active = progress * len(stages) >= index
        rounded(draw, (x, y, x + 408, y + 94), COLORS["panel_alt"], COLORS["cyan"] if active else COLORS["grid"])
        draw.text((x + 20, y + 16), name, font=FONTS["label"], fill=COLORS["cyan"] if active else COLORS["muted"])
        draw.text((x + 20, y + 51), detail, font=FONTS["body"], fill=COLORS["text"])
    draw.text(
        (42, 452),
        "Independent safe-region oracle checks post-event feasibility.",
        font=FONTS["body"],
        fill=COLORS["muted"],
    )


def scene_evidence(draw: ImageDraw.ImageDraw, progress: float) -> None:
    draw.text((38, 102), "Evidence, separated by tier", font=FONTS["title"], fill=COLORS["text"])
    tiers = [
        ("01", "Python contracts", "unit + statistical tests", COLORS["cyan"]),
        ("02", "C++ grid core", "strict compile + gtest", COLORS["cyan"]),
        ("03", "ROS 2 Jazzy / Nav2", "CI build + plugin discovery", COLORS["blue"]),
        ("04", "Gazebo execution", "static + frozen dynamic protocols", COLORS["blue"]),
        ("05", "Physical robot", "pending traceable evidence", COLORS["amber"]),
    ]
    for index, (number, title, detail, color) in enumerate(tiers):
        y = 164 + index * 62
        reached = progress * len(tiers) >= index
        draw.ellipse((44, y, 84, y + 40), fill=color if reached else COLORS["grid"])
        draw.text((54, y + 10), number, font=FONTS["small"], fill=COLORS["background"] if reached else COLORS["muted"])
        draw.text((106, y - 1), title, font=FONTS["label"], fill=COLORS["text"])
        draw.text((330, y - 1), detail, font=FONTS["body"], fill=COLORS["muted"])
        if index < len(tiers) - 1:
            draw.line((64, y + 40, 64, y + 62), fill=COLORS["grid"], width=3)
    rounded(draw, (700, 165, 910, 440), COLORS["panel"], COLORS["grid"])
    draw.text((730, 194), "No claim", font=FONTS["heading"], fill=COLORS["amber"])
    draw.text((730, 231), "moves upward", font=FONTS["body"], fill=COLORS["text"])
    draw.text((730, 260), "without an", font=FONTS["body"], fill=COLORS["text"])
    draw.text((730, 289), "artifact.", font=FONTS["body"], fill=COLORS["text"])
    draw.text((730, 350), "logs", font=FONTS["small"], fill=COLORS["muted"])
    draw.text((730, 373), "CSV / JSON", font=FONTS["small"], fill=COLORS["muted"])
    draw.text((730, 396), "SHA-256", font=FONTS["small"], fill=COLORS["muted"])


def scene_contributions(draw: ImageDraw.ImageDraw, progress: float) -> None:
    draw.text((38, 102), "26 contributions, one audit trail", font=FONTS["title"], fill=COLORS["text"])
    for index in range(26):
        col, row = index % 9, index // 9
        x, y = 42 + col * 99, 174 + row * 76
        active = index < max(1, int(progress * 27))
        rounded(draw, (x, y, x + 82, y + 54), COLORS["panel_alt"], COLORS["cyan"] if active else COLORS["grid"])
        draw.text(
            (x + 21, y + 16),
            f"C{index + 1:02d}",
            font=FONTS["label"],
            fill=COLORS["text"] if active else COLORS["muted"],
        )
    rounded(draw, (42, 414, 918, 474), COLORS["panel"], COLORS["grid"])
    draw.text(
        (66, 435),
        "registry  ->  executable command  ->  CSV artifact  ->  manifest + SHA-256",
        font=FONTS["body"],
        fill=COLORS["cyan"],
    )


def render_frames() -> list[Image.Image]:
    scenes = [
        scene_landing,
        scene_researcher,
        scene_research,
        scene_pipeline,
        scene_evidence,
        scene_contributions,
    ]
    frames: list[Image.Image] = []
    for scene_index, render in enumerate(scenes):
        for frame_index in range(FRAMES_PER_SCENE):
            progress = frame_index / (FRAMES_PER_SCENE - 1)
            image, draw = base_frame(scene_index, progress)
            render(draw, progress)
            frames.append(image)
    return frames


def write_gif(frames: list[Image.Image], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    quantized = [
        frame.resize(GIF_SIZE, Image.Resampling.LANCZOS).quantize(
            colors=64,
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
    with tempfile.TemporaryDirectory(prefix="dynnav-video-") as directory:
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
                "24",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-metadata",
                "comment=Technical overview; not recorded Gazebo or real-robot evidence",
                str(destination),
            ],
            check=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gif", type=Path, default=ROOT / "assets/dynnav_system_overview.gif")
    parser.add_argument("--mp4", type=Path, default=ROOT / "assets/dynnav_system_overview.mp4")
    args = parser.parse_args()
    frames = render_frames()
    write_gif(frames, args.gif)
    write_mp4(frames, args.mp4)
    print(f"Generated {len(frames)} frames: {args.gif} and {args.mp4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
