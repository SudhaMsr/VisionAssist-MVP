"""Shared constants and small helpers for the fine-tuning pipeline."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Same label preparation as the original training run (notebooks/vision-assist-v1.ipynb):
# polygons become enclosing boxes, invalid lines are skipped.
from scripts.evaluate_checkpoint import convert_line  # noqa: E402

CLASS_NAMES: list[str] = yaml.safe_load((REPO / "configs" / "rod_detection.yaml").read_text())["names"]
CLASS_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
CONFIG_DIR = REPO / "configs" / "finetune_v1"
SPLIT_DIR = CONFIG_DIR / "splits"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42

# COCO category -> ROD class. Every COCO category with a ROD equivalent is mapped, so those
# objects are never left unlabelled in COCO images. "bicycle" is deliberately NOT mapped: ROD's
# "Bike" mixes bicycles and motorbikes and is left unchanged in v1, so COCO images that contain a
# bicycle are skipped instead (an unlabelled bicycle would teach the model to ignore bikes).
COCO_TO_ROD = {
    "person": "Person", "car": "Car", "bus": "Bus", "truck": "Truck", "motorcycle": "Motorcycle",
    "dog": "Dog", "bench": "Bench", "chair": "Chair", "fire hydrant": "Fire hydrant", "potted plant": "Plant Pot",
}
COCO_EXCLUDE_IMAGES_WITH = {"bicycle"}
# ROD classes that COCO never labels; in COCO images these come only from label completion.
ROD_ONLY_CLASSES = [c for c in CLASS_NAMES if c not in set(COCO_TO_ROD.values())]
# Classes the COCO model adds to ROD images during label completion (people and vehicles only).
COMPLETE_IN_ROD = ["Person", "Car", "Bus", "Truck", "Motorcycle"]

# Classes whose detection quality must not drop (see docs and evaluate.py).
PROTECTED_CLASSES = ["Stairs", "Manhole", "Traffic Cone"]


def find_one(root: Path, relative: str) -> Path:
    """Find `relative` (e.g. 'coco2017/annotations/instances_train2017.json') anywhere under root."""
    root = Path(root)
    direct = root / relative
    if direct.exists():
        return direct
    name = Path(relative).name
    for dirpath, _, files in os.walk(root):
        if name in files and str(Path(dirpath, name)).endswith(relative):
            return Path(dirpath, name)
    raise FileNotFoundError(f"{relative} not found under {root}")


def read_list(path: Path) -> list[str]:
    """Lines of a list file, ignoring blanks and '#' comments."""
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def yolo_line(class_id: int, box_xyxy: tuple[float, float, float, float], w: int, h: int) -> str | None:
    """Pixel xyxy box -> normalised YOLO line, clipped to the image; None if degenerate."""
    x1, y1, x2, y2 = (max(0.0, min(float(v), lim)) for v, lim in zip(box_xyxy, (w, h, w, h)))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return (f"{class_id} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
            f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")


def read_yolo_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    """Normalised (class, cx, cy, w, h) boxes, with polygons converted like training does."""
    rows = []
    if not Path(path).exists():
        return rows
    for raw in Path(path).read_text().splitlines():
        if raw.strip():
            converted, _ = convert_line(raw)
            if converted:
                c, *v = converted.split()
                rows.append((int(c), *map(float, v)))
    return rows


def to_xyxy(row: tuple[int, float, float, float, float], w: int, h: int) -> tuple[float, float, float, float]:
    _, cx, cy, bw, bh = row
    return ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)


def overlap(a, b) -> tuple[float, float]:
    """(IoU, intersection over the smaller box) for two xyxy boxes."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    smaller = min(area_a, area_b)
    return (inter / union if union > 0 else 0.0, inter / smaller if smaller > 0 else 0.0)
