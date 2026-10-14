"""Build sample MP4s for the web UI from the ROD video-frame clips.

Encodes the ordered frame lists in configs/stream_*_clip.txt as H.264 MP4s in
outputs/samples/ (git-ignored), so "Choose an existing MP4" in the UI has real
footage. Needs the ROD dataset; existing files are never overwritten.

Usage (from the repo root):
    python scripts/make_sample_video.py [--images /path/to/ROD-Dataset/dataset] [--fps 2]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from vision_assist.config import read_image_list  # noqa: E402
from vision_assist.demo import find_ffmpeg  # noqa: E402
from vision_assist.stream import resolve_dataset_root  # noqa: E402

CLIPS = {"rod_people_clip.mp4": "configs/stream_people_clip.txt",
         "rod_dashcam_clip.mp4": "configs/stream_dashcam_clip.txt"}


def make_video(frames: list[Path], out: Path, fps: float, ffmpeg: str) -> None:
    first = cv2.imread(str(frames[0]))
    if first is None:
        raise SystemExit(f"Could not read {frames[0]}")
    h, w = first.shape[:2]
    proc = subprocess.Popen(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
         "-r", str(fps), "-i", "-", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
        stdin=subprocess.PIPE,
    )
    for path in frames:
        frame = cv2.imread(str(path))
        if frame is None:
            proc.kill()
            raise SystemExit(f"Could not read {path}")
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit(f"ffmpeg failed writing {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", help="ROD-Dataset/dataset folder (default: datasets/ or the Kaggle mount)")
    p.add_argument("--fps", type=float, default=2.0, help="Frame rate of the sample videos (default 2)")
    args = p.parse_args()
    root = resolve_dataset_root(args.images)
    out_dir = REPO / "outputs" / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_ffmpeg()
    for name, listing in CLIPS.items():
        out = out_dir / name
        if out.exists():
            print(f"exists, left unchanged: {out.relative_to(REPO)}")
            continue
        frames = read_image_list(REPO / listing, root)
        make_video(frames, out, args.fps, ffmpeg)
        print(f"wrote {out.relative_to(REPO)}: {len(frames)} frames at {args.fps:g} fps")


if __name__ == "__main__":
    main()
