"""Path resolution with no hard-coded Kaggle-only assumptions.

Precedence for every path: explicit argument > environment variable > default.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "rod_yolov8n_50ep_best.pt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "demo"

# ROD validation split unzipped into the repo (datasets/ is git-ignored).
LOCAL_IMAGE_DIR = REPO_ROOT / "datasets" / "ROD-Dataset" / "dataset" / "valid" / "images"

# Locations the ROD validation split has been mounted at on Kaggle. Only used
# as a fallback when no image directory is given.
KAGGLE_IMAGE_DIR_CANDIDATES = (
    Path("/kaggle/input/datasets/abtinzandi/obstacle-detection-dataset/ROD-Dataset/dataset/valid/images"),
    Path("/kaggle/input/obstacle-detection-dataset/ROD-Dataset/dataset/valid/images"),
)
DEFAULT_IMAGE_DIR_CANDIDATES = (LOCAL_IMAGE_DIR, *KAGGLE_IMAGE_DIR_CANDIDATES)

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
        candidates.extend(DEFAULT_IMAGE_DIR_CANDIDATES)
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No input images found. Pass --images DIR_OR_FILE, set VISION_ASSIST_IMAGES, or unzip the "
        "ROD dataset (kaggle.com/datasets/abtinzandi/obstacle-detection-dataset) into datasets/. "
        f"Tried: {', '.join(str(p) for p in candidates)}"
    )


DEFAULT_IMAGE_LIST = REPO_ROOT / "configs" / "demo_images.txt"


def read_image_list(list_file: str | os.PathLike, image_dir: Path) -> list[Path]:
    """Resolve a list of file names (one per line, `#` comments allowed) in `image_dir`."""
    names = [line.split("#", 1)[0].strip() for line in Path(list_file).read_text().splitlines()]
    names = [n for n in names if n]
    if not names:
        raise ValueError(f"No image names in {list_file}")
    missing = [n for n in names if not (image_dir / n).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} listed images not found in {image_dir}: {missing[:3]}")
    return [image_dir / n for n in names]


def list_images(source: Path, limit: int | None = None) -> list[Path]:
    """Return a sorted, deterministic list of images from a file or directory."""
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(
                f"Unsupported input {source.name}: only still images {sorted(IMAGE_EXTENSIONS)} are "
                "supported (no video or camera input)"
            )
        images = [source]
    else:
        images = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise FileNotFoundError(f"No images with extensions {sorted(IMAGE_EXTENSIONS)} in {source}")
    return images[:limit] if limit else images


DEFAULT_IMAGE_LIMIT = 8


def select_images(
    image_dir: Path, image_list: str | os.PathLike | None = None, limit: int | None = None
) -> tuple[list[Path], str]:
    """Pick the demo images and describe the choice.

    An explicit list wins, then an explicit limit. With neither, the curated
    `configs/demo_images.txt` is used if every listed image is in `image_dir`
    (the same default as the notebook), else the first DEFAULT_IMAGE_LIMIT.
    """
    if limit is not None and limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")
    if image_list:
        return read_image_list(image_list, image_dir), f"image list {image_list}"
    if limit is None and image_dir.is_dir():
        try:
            return read_image_list(DEFAULT_IMAGE_LIST, image_dir), f"curated list {DEFAULT_IMAGE_LIST.name}"
        except (FileNotFoundError, ValueError):
            pass
    n = limit or DEFAULT_IMAGE_LIMIT
    return list_images(image_dir, limit=n), f"first {n} images in sorted order"
