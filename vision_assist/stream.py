"""Continuous-stream alert timing: frames at real timestamps, one speech channel.

Usage:
    python -m vision_assist.stream --image-list configs/stream_people_clip.txt --fps 1
    python -m vision_assist.stream --video walk.mp4 --fps 2

The slideshow demo (vision_assist.demo) stretches every image to fit its speech,
so alerts can never collide. Here frames arrive at a fixed rate and speech takes
real time. While an alert is playing (plus a short gap) nothing new is spoken.
Nothing is queued either: when the channel is free again the *current* frame is
evaluated, so a hazard that has gone is never announced late.

One exception: an urgent (high-tier) alert that is ready to be spoken cuts off a
lower-tier message that is playing, so "Crosswalk" can never hold back "Car".
`--no-interrupt` turns this off.

The alert rules (prioritise, AlertScheduler) are unchanged. On busy frames a copy
of the scheduler is asked what it would have said, only to measure delays.
This is an offline simulation: speech is assumed to start at the frame's
timestamp, so processing time (reported separately) would add to every latency.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import wave
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from .config import DEFAULT_IMAGE_DIR_CANDIDATES, read_image_list, resolve_model_path
from .demo import _summary, _write_video, draw_banner, find_ffmpeg, probe_media
from .detector import Detector
from .guidance import HIGH, Alert, AlertScheduler, compose_message, prioritise
from .speech import assemble_timeline, find_tts_engine, synthesize, wav_duration

DEFAULT_STREAM_OUTPUT = Path(__file__).resolve().parent.parent / "outputs" / "stream"
OUTPUT_NAMES = ("vision_assist_stream.wav", "vision_assist_stream.mp4", "transcript.txt", "stream_results.json")
EPS = 1e-9


def _seconds(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {"n": len(values), "median_s": round(statistics.median(values), 2), "max_s": round(max(values), 2)}


@dataclass
class _Episode:
    """One continuous stretch of frames in which an alert key is present."""

    start_s: float
    tier: int
    announced: bool = False
    blocked: bool = False  # it would have been spoken but the channel was busy


class StreamTimer:
    """Decides, frame by frame, what is spoken on a single speech channel.

    `step` is called once per frame with that frame's prioritised alerts and a
    `speak(message) -> duration_s` callback. The scheduler is only consulted
    when the channel is free, so its rules and state are exactly those of the
    slideshow; busy frames use a throwaway copy for measurement only.
    """

    def __init__(self, scheduler: AlertScheduler, gap_s: float = 0.5, urgent_interrupts: bool = True):
        if gap_s < 0:
            raise ValueError("gap_s must be >= 0")
        self.scheduler = scheduler
        self.gap_s = gap_s
        self.urgent_interrupts = urgent_interrupts
        self.busy_until = 0.0
        self.events: list[dict] = []  # each spoken message: start_s, end_s, message, tier, interrupted
        self.spoken: list[dict] = []  # each spoken alert with its delays
        self.dropped: list[dict] = []  # qualified while busy, left view before it could be said
        self._pending: dict[tuple[str, str], float] = {}
        self._episodes: dict[tuple[str, str], _Episode] = {}
        self._closed_high: list[_Episode] = []

    def step(self, t: float, alerts: list[Alert], speak: Callable[[str], float]) -> dict:
        present = {a.key: a for a in alerts}
        for key in [k for k in self._episodes if k not in present]:
            self._close(key)
        for key, alert in present.items():
            self._episodes.setdefault(key, _Episode(start_s=t, tier=alert.tier))
        for key in [k for k in self._pending if k not in present]:
            self.dropped.append({"class_name": key[0], "image_region": key[1],
                                 "qualified_s": round(self._pending.pop(key), 2), "left_view_s": round(t, 2)})

        interrupted = None
        if t + EPS < self.busy_until:
            would, _ = copy.deepcopy(self.scheduler).select(alerts, now_s=t)
            playing = self.events[-1]
            if self.urgent_interrupts and playing["tier"] != HIGH and any(a.tier == HIGH for a in would):
                if t + EPS < playing["end_s"]:  # still talking, not just in the trailing gap: cut it off
                    playing["end_s"], playing["interrupted"] = t, True
                    interrupted = playing["message"]
            else:
                for alert in would:
                    self._pending.setdefault(alert.key, t)
                    self._episodes[alert.key].blocked = True
                return {"status": "busy", "message": None, "would_say": compose_message(would), "spoken": []}

        to_speak, suppressed = self.scheduler.select(alerts, now_s=t)
        message = compose_message(to_speak)
        if not message:
            return {"status": "quiet", "message": None, "would_say": None, "spoken": [],
                    "suppressed": [a.to_dict() for a in suppressed]}
        duration_s = speak(message)
        self.events.append({"start_s": t, "end_s": t + duration_s, "planned_end_s": t + duration_s,
                            "message": message, "tier": to_speak[0].tier, "interrupted": False})
        self.busy_until = t + duration_s + self.gap_s
        spoken = []
        for alert in to_speak:
            episode = self._episodes[alert.key]
            spoken.append({
                **alert.to_dict(),
                "queue_delay_s": round(t - self._pending.pop(alert.key, t), 2),
                "since_first_seen_s": round(t - episode.start_s, 2),
                "first_in_episode": not episode.announced,
            })
            episode.announced = True
        self.spoken.extend(spoken)
        return {"status": "spoken", "message": message, "would_say": None, "spoken": spoken,
                "suppressed": [a.to_dict() for a in suppressed], "interrupted": interrupted}

    def _close(self, key: tuple[str, str]) -> None:
        episode = self._episodes.pop(key)
        if episode.tier == HIGH:
            self._closed_high.append(episode)

    def metrics(self, stream_s: float) -> dict:
        for key in list(self._episodes):
            self._close(key)
        minutes = stream_s / 60
        speaking_s = sum(e["end_s"] - e["start_s"] for e in self.events)
        ordered = sorted(self.events, key=lambda e: e["start_s"])
        overlaps = sum(1 for a, b in zip(ordered, ordered[1:]) if b["start_s"] + EPS < a["end_s"])
        delayed = [s["queue_delay_s"] for s in self.spoken if s["queue_delay_s"] > 0]
        high = [s for s in self.spoken if s["tier"] == "high"]
        unannounced = [e for e in self._closed_high if not e.announced]
        return {
            "stream_duration_s": round(stream_s, 2),
            "messages_spoken": len(self.events),
            "alerts_spoken": len(self.spoken),
            "messages_per_min": round(len(self.events) / minutes, 1) if minutes else None,
            "alerts_per_min": round(len(self.spoken) / minutes, 1) if minutes else None,
            "speaking_s": round(speaking_s, 2),
            "speaking_pct": round(100 * speaking_s / stream_s, 1) if stream_s else None,
            "overlapping_messages": overlaps,
            "urgent_interruptions": sum(1 for e in self.events if e["interrupted"]),
            "speech_cut_s": round(sum(e["planned_end_s"] - e["end_s"] for e in self.events), 2),
            "queue_delay": {"alerts_delayed": len(delayed), **_seconds(delayed)},
            "dropped_while_busy": len(self.dropped),
            "still_waiting_at_end": len(self._pending),
            "urgent": {
                "high_tier_episodes": len(self._closed_high),
                "announced": len(self._closed_high) - len(unannounced),
                "first_announcement_latency": _seconds(
                    [s["since_first_seen_s"] for s in high if s["first_in_episode"]]
                ),
                "queue_delay_of_spoken": _seconds([s["queue_delay_s"] for s in high]),
                "not_announced_blocked_by_speech": sum(1 for e in unannounced if e.blocked),
                "not_announced_other": sum(1 for e in unannounced if not e.blocked),
            },
        }


def _truncate_wav(path: Path, seconds: float, fade_s: float = 0.02) -> None:
    """Cut a clip where it was interrupted, with a short fade-out so the cut does not click."""
    with wave.open(str(path), "rb") as w:
        params = w.getparams()
        data = w.readframes(w.getnframes())
    keep = int(seconds * params.framerate)  # round down so the clip still fits its slot
    data = data[: keep * params.sampwidth * params.nchannels]
    if params.sampwidth == 2:
        samples = np.frombuffer(data, dtype="<i2").astype(np.float64)
        n = min(len(samples), int(fade_s * params.framerate) * params.nchannels)
        if n:
            samples[-n:] *= np.linspace(1.0, 0.0, n)
        data = samples.astype("<i2").tobytes()
    with wave.open(str(path), "wb") as w:
        w.setparams(params)
        w.writeframes(data)


def frames_from_images(paths: list[Path], fps: float) -> Iterator[tuple[float, str, np.ndarray]]:
    for i, path in enumerate(paths):
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"Could not read image: {path}")
        yield i / fps, path.name, frame


def frames_from_video(path: Path, fps: float, limit: int | None = None) -> Iterator[tuple[float, str, np.ndarray]]:
    """Sample a video file at `fps`, taking the first source frame at or after each target time."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if not source_fps or source_fps <= 0:
        cap.release()
        raise ValueError(f"Video has no frame rate: {path}")
    try:
        taken = read = 0
        while limit is None or taken < limit:
            ok, frame = cap.read()
            if not ok:
                break
            source_t = read / source_fps
            read += 1
            if source_t + EPS >= taken / fps:
                yield taken / fps, f"{path.name}@{source_t:.2f}s", frame
                taken += 1
    finally:
        cap.release()


def run_stream(
    frames: Iterable[tuple[float, str, np.ndarray]],
    model_path: Path,
    out_dir: Path,
    fps: float,
    conf: float = 0.35,
    imgsz: int = 640,
    max_alerts: int = 2,
    cooldown_s: float = 6.0,
    gap_s: float = 0.5,
    tts_engine: list[str] | None = None,
    make_video: bool = True,
    device: str | None = None,
    urgent_interrupts: bool = True,
) -> dict:
    if not fps > 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    scheduler = AlertScheduler(max_alerts=max_alerts, cooldown_s=cooldown_s)
    timer = StreamTimer(scheduler, gap_s=gap_s, urgent_interrupts=urgent_interrupts)
    engine = tts_engine or find_tts_engine()
    ffmpeg = find_ffmpeg() if make_video else None
    detector = Detector(model_path, conf=conf, imgsz=imgsz, device=device)
    detector.detect(np.zeros((imgsz, imgsz, 3), dtype=np.uint8))  # warm-up

    out_dir = Path(out_dir)
    frames_dir, clips_dir = out_dir / "frames", out_dir / "speech_clips"
    for d in (frames_dir, clips_dir):
        d.mkdir(parents=True, exist_ok=True)
    for path in [*frames_dir.glob("frame_*.jpg"), *clips_dir.glob("alert_*.wav"), *(out_dir / n for n in OUTPUT_NAMES)]:
        path.unlink(missing_ok=True)

    records, frame_paths, clip_paths = [], [], []
    inference_ms, tts_ms = [], []
    for index, (t, source, frame) in enumerate(frames, start=1):
        result = detector.detect(frame)

        def speak(message: str, index: int = index) -> float:
            clip = clips_dir / f"alert_{index:04d}.wav"
            tts_ms.append(synthesize(message, clip, engine))
            clip_paths.append(clip)
            return wav_duration(clip)

        outcome = timer.step(t, prioritise(result.detections), speak)
        inference_ms.append(result.inference_ms)

        caption = [f"[{index}] t={t:.1f}s  {source}"]
        if outcome["status"] == "spoken" and outcome["interrupted"]:
            caption.append(f"CUTS OFF '{outcome['interrupted']}' -> SPEAKING: {outcome['message']}")
        elif outcome["status"] == "spoken":
            caption.append(f"SPEAKING: {outcome['message']}")
        elif outcome["status"] == "busy":
            caption.append("(still speaking)" + (f" would say: {outcome['would_say']}" if outcome["would_say"] else ""))
        else:
            caption.append("SPOKEN: (nothing)")
        frame_path = frames_dir / f"frame_{index:04d}.jpg"
        if not cv2.imwrite(str(frame_path), draw_banner(result.annotated, caption)):
            raise RuntimeError(f"Could not write {frame_path}")
        frame_paths.append(frame_path)
        records.append({
            "index": index, "t_s": round(t, 3), "source": source, **outcome,
            "detections": [d.to_dict() for d in result.detections],
            "inference_ms": round(result.inference_ms, 1),
        })
    if not records:
        raise ValueError("No frames to process")

    period = 1 / fps
    stream_s = len(records) * period
    events = timer.events
    for event, clip in zip(events, clip_paths):
        if event["interrupted"]:
            _truncate_wav(clip, event["end_s"] - event["start_s"])
    total_s = max(stream_s, max((e["end_s"] for e in events), default=0.0))
    # One silent lead-in, then each clip held until the next one starts (never overlapping).
    starts = [e["start_s"] for e in events] + [total_s]
    clips: list[Path | None] = [None, *clip_paths]
    slots = [starts[0]] + [b - a for a, b in zip(starts, starts[1:])]
    audio_path = out_dir / OUTPUT_NAMES[0]
    audio_s = assemble_timeline(clips, slots, audio_path)

    video_path, video_info = out_dir / OUTPUT_NAMES[1], None
    if make_video:
        durations = [period] * len(frame_paths)
        durations[-1] += total_s - stream_s  # hold the last frame while the final alert finishes
        _write_video(frame_paths, durations, audio_path, video_path, ffmpeg)
        video_info = probe_media(video_path, ffmpeg)

    (out_dir / "transcript.txt").write_text(
        "".join(f"{e['start_s']:7.1f}s  {e['message']}{' [cut off]' if e['interrupted'] else ''}\n" for e in events),
        encoding="utf-8",
    )
    summary = {
        "vision_assist_version": __version__,
        "model": Path(model_path).name,
        "settings": {"fps": fps, "conf": conf, "imgsz": imgsz, "max_alerts": max_alerts,
                     "cooldown_s": cooldown_s, "gap_s": gap_s, "urgent_interrupts": urgent_interrupts,
                     "device": device or "auto"},
        "frames": len(records),
        "metrics": timer.metrics(stream_s),
        "processing_ms": {"frame_period_ms": round(1000 * period, 1), "inference": _summary(inference_ms),
                          "tts": _summary(tts_ms)},
        "outputs": {"audio": audio_path.name, "audio_duration_s": round(audio_s, 2),
                    "video": video_path.name if make_video else None, "video_probe": video_info},
        "dropped_while_busy": timer.dropped,
        "per_frame": records,
    }
    (out_dir / "stream_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def resolve_dataset_root(explicit: str | None = None) -> Path:
    """Folder that stream clip lists are relative to: the ROD `dataset/` folder."""
    candidates = [Path(explicit)] if explicit else [p.parents[1] for p in DEFAULT_IMAGE_DIR_CANDIDATES]
    for path in candidates:
        if path.is_dir():
            return path
    raise FileNotFoundError(
        "ROD dataset folder not found. Pass --images DIR or unzip ROD into datasets/. "
        f"Tried: {', '.join(str(p) for p in candidates)}"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="Video file to sample at --fps")
    src.add_argument("--image-list", help="Ordered frame list, paths relative to --images "
                     "(e.g. configs/stream_people_clip.txt)")
    p.add_argument("--images", help="Folder the --image-list paths are relative to "
                   "(default: datasets/ROD-Dataset/dataset or the Kaggle mount)")
    p.add_argument("--fps", type=float, default=1.0, help="Frames per second of the stream (default 1)")
    p.add_argument("--limit", type=int, default=None, help="Process at most N frames")
    p.add_argument("--model", help="Checkpoint path (default: models/rod_yolov8n_50ep_best.pt)")
    p.add_argument("--out", help="Output directory (default: outputs/stream)")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--max-alerts", type=int, default=2, help="Max objects named per message")
    p.add_argument("--cooldown", type=float, default=6.0, help="Seconds before repeating the same alert")
    p.add_argument("--gap", type=float, default=0.5, help="Silence after each message before the next (s)")
    p.add_argument("--device", default=None, help="e.g. cpu, 0 (default: auto)")
    p.add_argument("--tts", default=None, help="TTS executable (default: espeak-ng / $VISION_ASSIST_TTS)")
    p.add_argument("--no-video", action="store_true", help="Skip MP4 export")
    p.add_argument("--no-interrupt", action="store_true",
                   help="Never cut off a message, even for an urgent alert (the earlier behaviour)")
    args = p.parse_args(argv)

    out = Path(args.out) if args.out else DEFAULT_STREAM_OUTPUT
    try:
        if args.limit is not None and args.limit < 1:
            raise ValueError(f"limit must be >= 1, got {args.limit}")
        if not args.fps > 0:
            raise ValueError(f"fps must be > 0, got {args.fps}")
        if args.video:
            video = Path(args.video)
            if not video.is_file():
                raise FileNotFoundError(f"Video not found: {video}")
            frames = frames_from_video(video, args.fps, args.limit)
            source = str(video)
        else:
            paths = read_image_list(args.image_list, resolve_dataset_root(args.images))[: args.limit]
            unreadable = [str(q) for q in paths if cv2.imread(str(q)) is None]
            if unreadable:
                raise ValueError(f"{len(unreadable)} frame(s) could not be read: {unreadable[:3]}")
            frames = frames_from_images(paths, args.fps)
            source = f"{len(paths)} frames from {args.image_list}"
        print(f"Stream: {source} at {args.fps:g} fps", flush=True)
        summary = run_stream(
            frames, resolve_model_path(args.model), out, fps=args.fps, conf=args.conf, imgsz=args.imgsz,
            max_alerts=args.max_alerts, cooldown_s=args.cooldown, gap_s=args.gap,
            tts_engine=find_tts_engine(args.tts), make_video=not args.no_video, device=args.device,
            urgent_interrupts=not args.no_interrupt,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print((out / "transcript.txt").read_text(), end="")
    print(f"\nOutputs written to {out}")
    print(json.dumps(summary["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
