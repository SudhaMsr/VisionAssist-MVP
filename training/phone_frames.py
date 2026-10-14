"""Extract frames from phone walk videos for labelling (see docs/phone_footage_spec.md).

Each video is one walk. Frames are sampled at --fps (default 1 per second, so
consecutive frames are not near-duplicates) and written as
<out>/<walk>/images/<walk>_<NNNN>.jpg, ready to import into CVAT or Label Studio.
The walk name is the video file name (letters, digits, - and _ only), e.g. walk01_high_street.mp4.

Usage:
    python -m training.phone_frames --videos ~/walks --out ~/phone_dataset [--fps 1]
Existing walk folders are skipped, never overwritten.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2

WALK_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


def extract(video: Path, out_root: Path, fps: float = 1.0) -> dict:
    walk = video.stem
    if not WALK_NAME.match(walk):
        raise ValueError(f"Rename {video.name}: use only letters, digits, - and _ (e.g. walk01_high_street.mp4)")
    images = out_root / walk / "images"
    if images.exists():
        return {"walk": walk, "status": "exists, skipped"}
    from vision_assist.stream import frames_from_video

    images.mkdir(parents=True)
    (out_root / walk / "labels").mkdir()
    n, size = 0, None
    for _, _, frame in frames_from_video(video, fps):
        n += 1
        size = f"{frame.shape[1]}x{frame.shape[0]}"
        if not cv2.imwrite(str(images / f"{walk}_{n:04d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"Could not write frame {n} of {video.name}")
    return {"walk": walk, "frames": n, "size": size, "status": "written"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos", required=True, help="Folder of .mp4 walk videos")
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=float, default=1.0)
    args = p.parse_args(argv)
    videos = sorted(v for v in Path(args.videos).iterdir() if v.suffix.lower() == ".mp4")
    if not videos:
        raise SystemExit(f"No .mp4 files in {args.videos}")
    results = [extract(v, Path(args.out), args.fps) for v in videos]
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
