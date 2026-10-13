"""Offline text-to-speech via eSpeak NG, plus WAV timeline assembly.

The engine is a command prefix (list of strings) so tests can substitute a
fake engine that accepts the same arguments.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import wave
from dataclasses import dataclass
from pathlib import Path


class SpeechError(RuntimeError):
    pass


class TTSUnavailableError(SpeechError):
    pass


def find_tts_engine(explicit: str | None = None) -> list[str]:
    candidate = explicit or os.environ.get("VISION_ASSIST_TTS")
    if candidate:
        resolved = shutil.which(candidate)
        if resolved is None:
            raise TTSUnavailableError(f"TTS engine not found or not executable: {candidate}")
        return [resolved]
    for name in ("espeak-ng", "espeak"):
        resolved = shutil.which(name)
        if resolved:
            return [resolved]
    raise TTSUnavailableError(
        "eSpeak NG not found. Install it (Debian/Ubuntu/Kaggle: apt-get install -y espeak-ng; "
        "macOS: brew install espeak-ng) or set VISION_ASSIST_TTS to the executable."
    )


@dataclass(frozen=True)
class AudioFormat:
    nchannels: int
    sampwidth: int
    framerate: int


def synthesize(
    text: str,
    out_path: Path,
    engine: list[str],
    voice: str = "en",
    words_per_minute: int = 150,
    timeout_s: float = 30.0,
) -> float:
    """Write `text` as speech to `out_path`. Returns synthesis time in ms."""
    if not text or not text.strip():
        raise ValueError("Cannot synthesise empty text")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [*engine, "-v", voice, "-s", str(words_per_minute), "-w", str(out_path), text]
    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise SpeechError(f"TTS timed out after {timeout_s}s") from exc
    elapsed_ms = (time.perf_counter() - start) * 1000
    if proc.returncode != 0:
        raise SpeechError(f"TTS failed (exit {proc.returncode}): {proc.stderr.strip()[-500:]}")
    if not out_path.is_file() or wav_duration(out_path) <= 0:
        raise SpeechError(f"TTS produced no audio: {out_path}")
    return elapsed_ms


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def assemble_timeline(
    clips: list[Path | None],
    slot_durations: list[float],
    out_path: Path,
    fmt: AudioFormat | None = None,
) -> float:
    """Place each clip at the start of its slot, pad with silence, write one WAV.

    Every clip must fit in its slot (the caller sizes slots from clip lengths).
    Returns total duration in seconds.
    """
    if len(clips) != len(slot_durations):
        raise ValueError("clips and slot_durations must have the same length")

    loaded: list[bytes | None] = []
    for clip in clips:
        if clip is None:
            loaded.append(None)
            continue
        with wave.open(str(clip), "rb") as w:
            clip_fmt = AudioFormat(w.getnchannels(), w.getsampwidth(), w.getframerate())
            data = w.readframes(w.getnframes())
        if fmt is None:
            fmt = clip_fmt
        elif clip_fmt != fmt:
            raise SpeechError(f"Incompatible audio format in {clip}: {clip_fmt} != {fmt}")
        loaded.append(data)
    if fmt is None:  # no speech at all: silent track
        fmt = AudioFormat(nchannels=1, sampwidth=2, framerate=22050)

    bytes_per_frame = fmt.nchannels * fmt.sampwidth
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_frames = 0
    with wave.open(str(out_path), "wb") as out:
        out.setnchannels(fmt.nchannels)
        out.setsampwidth(fmt.sampwidth)
        out.setframerate(fmt.framerate)
        for data, slot in zip(loaded, slot_durations):
            slot_frames = round(slot * fmt.framerate)
            data = data or b""
            clip_frames = len(data) // bytes_per_frame
            if clip_frames > slot_frames:
                raise ValueError(f"Clip of {clip_frames} frames does not fit slot of {slot_frames}")
            out.writeframes(data + b"\x00" * bytes_per_frame * (slot_frames - clip_frames))
            total_frames += slot_frames
    return total_frames / fmt.framerate
