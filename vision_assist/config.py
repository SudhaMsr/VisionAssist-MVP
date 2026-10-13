"""Path resolution with no hard-coded Kaggle-only assumptions.

Precedence for every path: explicit argument > environment variable > default.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "rod_yolov8n_50ep_best.pt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "demo"

# Locations the ROD validation split has been mounted at on Kaggle. Only used
# as a fallback when no image directory is given.
KAGGLE_IMAGE_DIR_CANDIDATES = (
    Path("/kaggle/input/datasets/abtinzandi/obstacle-detection-dataset/ROD-Dataset/dataset/valid/images"),
    Path("/kaggle/input/obstacle-detection-dataset/ROD-Dataset/dataset/valid/images"),
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve_model_path(explicit: str | os.PathLike | None = None) -> Path:
    path = Path(explicit or os.environ.get("VISION_ASSIST_MODEL") or DEFAULT_MODEL_PATH)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {path}. Pass --model or set VISION_ASSIST_MODEL."
        )
    return path


def resolve_output_dir(explicit: str | os.PathLike | None = None) -> Path:
    return Path(explicit or os.environ.get("VISION_ASSIST_OUTPUT") or DEFAULT_OUTPUT_DIR)


def resolve_image_dir(explicit: str | os.PathLike | None = None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    elif os.environ.get("VISION_ASSIST_IMAGES"):
        candidates.append(Path(os.environ["VISION_ASSIST_IMAGES"]))
    else:
        candidates.extend(KAGGLE_IMAGE_DIR_CANDIDATES)
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No input images found. Pass --images DIR_OR_FILE or set VISION_ASSIST_IMAGES. "
        f"Tried: {', '.join(str(p) for p in candidates)}"
    )


def list_images(source: Path, limit: int | None = None) -> list[Path]:
    """Return a sorted, deterministic list of images from a file or directory."""
    if source.is_file():
        images = [source]
    else:
        images = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise FileNotFoundError(f"No images with extensions {sorted(IMAGE_EXTENSIONS)} in {source}")
    return images[:limit] if limit else images
