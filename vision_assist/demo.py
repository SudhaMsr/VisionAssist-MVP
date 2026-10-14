"""Reproducible image-sequence demo: detect -> prioritise -> speak -> MP4.

Usage:
    python -m vision_assist.demo [--images PATH] [--image-list FILE | --limit N] [--out DIR]

Treats a list of still images as a slideshow. It is not a camera pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from .config import resolve_image_dir, resolve_model_path, resolve_output_dir, select_images
from .detector import Detector
from .guidance import AlertScheduler, compose_message, prioritise
from .speech import assemble_timeline, find_tts_engine, synthesize, wav_duration

VIDEO_FPS = 10


def find_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # ImportError or missing bundled binary
        raise RuntimeError(
            "ffmpeg not found. Install ffmpeg or `pip install imageio-ffmpeg`, or run with --no-video."
        ) from exc


def probe_media(path: Path, ffmpeg: str) -> dict:
    """Parse `ffmpeg -i` output: duration and whether audio/video streams exist."""
    proc = subprocess.run([ffmpeg, "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    info = proc.stderr
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", info)
    duration = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]) if m else None
    return {
        "duration_s": duration,
        "has_video": "Video:" in info,
        "has_audio": "Audio:" in info,
    }


def draw_banner(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """Draw caption lines on a dark strip below the frame (does not cover detections)."""
    h, w = frame.shape[:2]
    scale = max(0.5, w / 1400)
    thickness = max(1, round(scale * 2))
    line_h = int(32 * scale) + 8
    wrap_chars = max(20, int(w / (17 * scale)))
    wrapped = [seg for line in lines for seg in textwrap.wrap(line, wrap_chars)] or [""]
    strip = np.zeros((line_h * len(wrapped) + 12, w, 3), dtype=np.uint8)
    for i, text in enumerate(wrapped):
        y = line_h * (i + 1)
        cv2.putText(strip, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return np.vstack([frame, strip])


def _quantise(seconds: float) -> float:
    """Round up to a whole number of video frames so audio and video stay aligned."""
    return math.ceil(seconds * VIDEO_FPS) / VIDEO_FPS


def _summary(values: list[float]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]
    return {
        "n": len(values),
        "mean_ms": round(statistics.fmean(values), 1),
        "median_ms": round(statistics.median(values), 1),
        "p95_ms": round(p95, 1),
        "max_ms": round(max(values), 1),
    }


def run_demo(
    images: list[Path],
    model_path: Path,
    out_dir: Path,
    conf: float = 0.35,
    imgsz: int = 640,
    max_alerts: int = 2,
    cooldown_s: float = 6.0,
    min_frame_s: float = 2.0,
    gap_s: float = 0.5,
    tts_engine: list[str] | None = None,
    make_video: bool = True,
    device: str | None = None,
) -> dict:
    if not images:
        raise ValueError("No images to process")
    # Check everything that can fail before writing any output, so a bad input
    # never leaves a half-finished demo behind.
    scheduler = AlertScheduler(max_alerts=max_alerts, cooldown_s=cooldown_s)
    unreadable = [p for p in images if cv2.imread(str(p)) is None]
    if unreadable:
        raise ValueError(f"{len(unreadable)} input image(s) could not be read: {[str(p) for p in unreadable[:3]]}")
    engine = tts_engine or find_tts_engine()
    ffmpeg = find_ffmpeg() if make_video else None  # fail early, before inference

    detector = Detector(model_path, conf=conf, imgsz=imgsz, device=device)
    # Warm-up so the first timed inference doesn't include one-off setup cost.
    detector.detect(np.zeros((imgsz, imgsz, 3), dtype=np.uint8))

    out_dir = Path(out_dir)
    frames_dir = out_dir / "frames"
    clips_dir = out_dir / "speech_clips"
    for d in (frames_dir, clips_dir):
        d.mkdir(parents=True, exist_ok=True)
    # Remove this tool's files from an earlier run, so a new run (even one that
    # fails part-way) is never mixed with leftovers. Other files are left alone.
    stale = [*frames_dir.glob("frame_*.jpg"), *clips_dir.glob("alert_*.wav")]
    stale += [out_dir / name for name in (
        "vision_assist_demo.wav", "vision_assist_demo.mp4", "transcript.txt", "demo_results.json",
    )]
    for path in stale:
        path.unlink(missing_ok=True)

    timeline_s = 0.0
    records, clips, slots, frame_paths = [], [], [], []
    inference_ms, tts_ms, end_to_end_ms = [], [], []

    for index, image_path in enumerate(images, start=1):
        t0 = time.perf_counter()
        result = detector.detect(image_path)
        alerts = prioritise(result.detections)
        spoken, suppressed = scheduler.select(alerts, now_s=timeline_s)
        message = compose_message(spoken)

        clip_path, clip_s = None, 0.0
        if message:
            clip_path = clips_dir / f"alert_{index:02d}.wav"
            tts_ms.append(synthesize(message, clip_path, engine))
            clip_s = wav_duration(clip_path)
        end_to_end_ms.append((time.perf_counter() - t0) * 1000)
        inference_ms.append(result.inference_ms)

        slot_s = _quantise(max(min_frame_s, clip_s + gap_s))
        caption = [f"[{index}/{len(images)}] {image_path.name}"]
        caption.append(f"SPOKEN: {message}" if message else "SPOKEN: (nothing)")
        if suppressed:
            caption.append("Not spoken: " + "; ".join(a.phrase for a in suppressed))
        frame_path = frames_dir / f"frame_{index:02d}.jpg"
        if not cv2.imwrite(str(frame_path), draw_banner(result.annotated, caption)):
            raise RuntimeError(f"Could not write {frame_path}")

        records.append({
            "index": index,
            "image": image_path.name,
            "start_s": round(timeline_s, 2),
            "duration_s": slot_s,
            "detections": [d.to_dict() for d in result.detections],
            "alerts_spoken": [a.to_dict() for a in spoken],
            "alerts_suppressed": [a.to_dict() for a in suppressed],
            "message": message,
            "inference_ms": round(result.inference_ms, 1),
        })
        clips.append(clip_path)
        slots.append(slot_s)
        frame_paths.append(frame_path)
        timeline_s += slot_s
        print(f"{index}/{len(images)} {image_path.name}: {message or '(no alert)'}", flush=True)

    audio_path = out_dir / "vision_assist_demo.wav"
    audio_s = assemble_timeline(clips, slots, audio_path)

    video_info = None
    video_path = out_dir / "vision_assist_demo.mp4"
    if make_video:
        _write_video(frame_paths, slots, audio_path, video_path, ffmpeg)
        video_info = probe_media(video_path, ffmpeg)

    (out_dir / "transcript.txt").write_text(
        "\n".join(f"{r['start_s']:6.1f}s  {r['image']}: {r['message'] or '(no alert)'}" for r in records) + "\n",
        encoding="utf-8",
    )

    try:
        import torch

        hardware = "cuda:" + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        hardware = "unknown"
    summary = {
        "vision_assist_version": __version__,
        "model": model_path.name,
        "settings": {
            "conf": conf, "imgsz": imgsz, "max_alerts": max_alerts, "cooldown_s": cooldown_s,
            "min_frame_s": min_frame_s, "gap_s": gap_s, "device": device or "auto",
        },
        "hardware": hardware,
        "image_count": len(images),
        "latency": {
            "model_load_ms": round(detector.load_ms, 1),
            "inference": _summary(inference_ms),
            "tts": _summary(tts_ms),
            "per_frame_detect_to_audio": _summary(end_to_end_ms),
        },
        "outputs": {
            "audio": audio_path.name,
            "audio_duration_s": round(audio_s, 2),
            "video": video_path.name if make_video else None,
            "video_probe": video_info,
        },
        "frames": records,
    }
    (out_dir / "demo_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _write_video(frame_paths, slots, audio_path, video_path, ffmpeg) -> None:
    first = cv2.imread(str(frame_paths[0]))
    height, width = first.shape[:2]
    silent = video_path.with_name(video_path.stem + "_silent.mp4")
    writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, (width, height))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open a video writer")
    try:
        for frame_path, slot in zip(frame_paths, slots):
            frame = cv2.imread(str(frame_path))
            if frame.shape[:2] != (height, width):
                # Letterbox into the first frame's size to keep aspect ratio.
                s = min(width / frame.shape[1], height / frame.shape[0])
                resized = cv2.resize(frame, (int(frame.shape[1] * s), int(frame.shape[0] * s)))
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                frame[: resized.shape[0], : resized.shape[1]] = resized
            for _ in range(round(slot * VIDEO_FPS)):
                writer.write(frame)
    finally:
        writer.release()
    proc = subprocess.run(
        [
            ffmpeg, "-y", "-loglevel", "error",
            "-i", str(silent), "-i", str(audio_path),
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            "-shortest", str(video_path),
        ],
        capture_output=True, text=True,
    )
    silent.unlink(missing_ok=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()[-800:]}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", help="Image file or directory (default: $VISION_ASSIST_IMAGES, then ROD valid/images "
                   "under datasets/ or on Kaggle)")
    p.add_argument("--model", help="Checkpoint path (default: models/rod_yolov8n_50ep_best.pt)")
    p.add_argument("--out", help="Output directory (default: outputs/demo)")
    p.add_argument("--image-list", help="Text file of image names inside --images, in playback order; overrides "
                   "--limit (default: configs/demo_images.txt if all its images are present)")
    p.add_argument("--limit", type=int, default=None, help="Use the first N images in sorted order (default 8 "
                   "when the curated list does not apply)")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--max-alerts", type=int, default=2, help="Max objects named per frame")
    p.add_argument("--cooldown", type=float, default=6.0, help="Seconds before repeating the same alert")
    p.add_argument("--device", default=None, help="e.g. cpu, 0 (default: auto)")
    p.add_argument("--tts", default=None, help="TTS executable (default: espeak-ng / $VISION_ASSIST_TTS)")
    p.add_argument("--no-video", action="store_true", help="Skip MP4 export")
    args = p.parse_args(argv)

    try:
        image_dir = resolve_image_dir(args.images)
        images, chosen = select_images(image_dir, args.image_list, args.limit)
        print(f"Images: {len(images)} from {image_dir} ({chosen})", flush=True)
        summary = run_demo(
            images=images,
            model_path=resolve_model_path(args.model),
            out_dir=resolve_output_dir(args.out),
            conf=args.conf,
            imgsz=args.imgsz,
            max_alerts=args.max_alerts,
            cooldown_s=args.cooldown,
            tts_engine=find_tts_engine(args.tts),
            make_video=not args.no_video,
            device=args.device,
        )
    except (OSError, ValueError, RuntimeError) as exc:  # OSError covers missing files and unwritable --out
        print(f"error: {exc}", file=sys.stderr)
        return 1
    out = resolve_output_dir(args.out)
    print(f"\nOutputs written to {out}")
    print(json.dumps(summary["latency"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
