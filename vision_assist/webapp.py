"""Helpers for the Streamlit UI (app.py). No Streamlit imports, so they are unit-testable.

The UI analyses an MP4 with the existing stream pipeline (vision_assist.stream):
the same detector, alert rules, single speech channel and urgent interruption.
Every run writes to a new folder, so earlier runs and results/demo/ are never touched.
"""

from __future__ import annotations

import io
import os
import re
import zipfile
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .config import REPO_ROOT, resolve_model_path
from .detector import Detection
from .guidance import HAZARD_TIERS, IGNORED_CLASSES, MEDIUM, TIER_LABELS, image_region
from .speech import find_tts_engine
from .stream import frames_from_video, run_stream

SAMPLE_VIDEO_DIR = REPO_ROOT / "outputs" / "samples"
UI_OUTPUT_DIR = REPO_ROOT / "outputs" / "ui_runs"


def video_dirs() -> list[Path]:
    """Folders offered in "Choose a video": $VISION_ASSIST_VIDEOS, else outputs/samples/."""
    env = os.environ.get("VISION_ASSIST_VIDEOS")
    return [Path(p) for p in env.split(os.pathsep) if p] if env else [SAMPLE_VIDEO_DIR]


def ui_output_root() -> Path:
    return Path(os.environ.get("VISION_ASSIST_UI_OUTPUT") or UI_OUTPUT_DIR)


def list_videos(dirs: list[Path] | None = None) -> list[Path]:
    found = []
    for d in dirs if dirs is not None else video_dirs():
        if d.is_dir():
            found += [p for p in d.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"]
    return sorted(found)


def new_run_dir(video_name: str, root: Path | None = None) -> Path:
    """A fresh, never-reused folder for one analysis."""
    root = root or ui_output_root()
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(video_name).stem).strip("_")[:40] or "video"
    base = root / f"{datetime.now():%Y%m%d-%H%M%S}-{stem}"
    path, n = base, 1
    while path.exists():
        n += 1
        path = base.with_name(f"{base.name}-{n}")
    path.mkdir(parents=True)
    return path


def save_upload(name: str, data: bytes, run_dir: Path) -> Path:
    if Path(name).suffix.lower() != ".mp4":
        raise ValueError(f"Please upload an .mp4 file (got {name!r})")
    if not data:
        raise ValueError("The uploaded file is empty")
    path = run_dir / "input.mp4"
    path.write_bytes(data)
    return path


def expected_frames(video: Path, fps: float, limit: int | None) -> int:
    """How many frames `frames_from_video` will yield (for a progress bar)."""
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video.name}. Is it a valid MP4?")
        source_fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    finally:
        cap.release()
    if not source_fps or source_fps <= 0 or count <= 0:
        raise ValueError(f"Could not read the frame rate or length of {video.name}")
    n = int(np.ceil(count / source_fps * fps - 1e-9))
    return max(1, min(n, limit) if limit else n)


def _with_progress(frames: Iterator, total: int, progress: Callable[[int, int], None] | None) -> Iterator:
    for i, item in enumerate(frames, start=1):
        if progress:
            progress(min(i, total), total)
        yield item


def analyse_video(
    video: Path,
    out_dir: Path,
    fps: float = 2.0,
    limit: int | None = 120,
    conf: float = 0.35,
    max_alerts: int = 2,
    cooldown_s: float = 6.0,
    urgent_interrupts: bool = True,
    progress: Callable[[int, int], None] | None = None,
    device: str | None = "cpu",
) -> dict:
    """Run the existing stream pipeline on one video. Raises with a readable message on failure."""
    if not fps > 0:
        raise ValueError("Frames per second must be above 0")
    if limit is not None and limit < 1:
        raise ValueError("Maximum frames must be at least 1")
    model = resolve_model_path(None)
    engine = find_tts_engine()  # fail before any work if espeak-ng is missing
    total = expected_frames(video, fps, limit)
    frames = _with_progress(frames_from_video(video, fps, limit), total, progress)
    return run_stream(
        frames, model, out_dir, fps=fps, conf=conf, max_alerts=max_alerts, cooldown_s=cooldown_s,
        tts_engine=engine, device=device, urgent_interrupts=urgent_interrupts,
    )


def alert_rows(summary: dict) -> list[dict]:
    """One row per spoken message, in time order."""
    rows = []
    frames = summary["per_frame"]
    for i, f in enumerate(frames):
        if f["status"] != "spoken":
            continue
        cut = next((g for g in frames[i + 1:] if g["status"] == "spoken"), None)
        rows.append({
            "time_s": f["t_s"],
            "frame": f["index"],
            "message": f["message"],
            "priority": "urgent" if any(a["tier"] == "high" for a in f["spoken"]) else "normal",
            "cut off later": bool(cut and cut.get("interrupted") == f["message"]),
            "cut off what was playing": f.get("interrupted") or "",
        })
    return rows


def detection_rows(summary: dict) -> list[dict]:
    rows = []
    for f in summary["per_frame"]:
        for d in f["detections"]:
            det = Detection(d["class_name"], d["confidence"], tuple(d["bbox"]), d["image_width"], d["image_height"])
            tier = "ignored" if det.class_name in IGNORED_CLASSES else TIER_LABELS[
                HAZARD_TIERS.get(det.class_name, MEDIUM)]
            rows.append({
                "frame": f["index"], "time_s": f["t_s"], "class": det.class_name, "tier": tier,
                "confidence": round(det.confidence, 2), "image region": image_region(det),
                "box area %": round(100 * det.area_frac, 1), "status": f["status"],
            })
    return rows


def frame_caption(frame: dict) -> str:
    if frame["status"] == "spoken":
        said = f"Spoke: {frame['message']}"
        return f"{said} (cut off: {frame['interrupted']})" if frame.get("interrupted") else said
    if frame["status"] == "busy":
        return "Still speaking" + (f"; held back: {frame['would_say']}" if frame.get("would_say") else "")
    return "Nothing new to say"


def zip_outputs(run_dir: Path) -> bytes:
    """All outputs of one run (minus the uploaded input) as a ZIP."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(run_dir.rglob("*")):
            if path.is_file() and path.name != "input.mp4":
                z.write(path, path.relative_to(run_dir))
    return buf.getvalue()
