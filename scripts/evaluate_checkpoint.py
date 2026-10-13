"""Evaluate the existing checkpoint on one ROD split (default: test).

Reproduces the label preparation from notebooks/vision-assist-v1.ipynb (cell 2):
polygon labels become enclosing boxes, invalid lines are counted and skipped.
Images are symlinked; the source dataset is never modified.

Usage:
    python scripts/evaluate_checkpoint.py --dataset /path/to/ROD-Dataset/dataset [--split test]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from vision_assist.config import IMAGE_EXTENSIONS, resolve_model_path  # noqa: E402

CLASS_NAMES = yaml.safe_load((REPO / "configs" / "rod_detection.yaml").read_text())["names"]


def convert_line(line: str) -> tuple[str | None, str]:
    """Return (yolo_bbox_line or None, kind) where kind is bbox|polygon|skipped."""
    parts = line.split()
    try:
        class_id = int(parts[0])
        coords = [float(x) for x in parts[1:]]
    except (ValueError, OverflowError, IndexError):
        return None, "skipped"
    if not 0 <= class_id < len(CLASS_NAMES):
        return None, "skipped"
    if len(coords) == 4:
        x, y, w, h = coords
        kind = "bbox"
    elif len(coords) >= 6 and len(coords) % 2 == 0:
        xs, ys = coords[0::2], coords[1::2]
        x, y = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        kind = "polygon"
    else:
        return None, "skipped"
    if not all(0 <= v <= 1 for v in (x, y, w, h)) or w <= 0 or h <= 0:
        return None, "skipped"
    return f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}", kind


def prepare_split(source: Path, split: str, work: Path) -> dict:
    src_images, src_labels = source / split / "images", source / split / "labels"
    if not src_images.is_dir() or not src_labels.is_dir():
        raise FileNotFoundError(f"Missing images/labels for split '{split}' under {source}")
    out_images, out_labels = work / split / "images", work / split / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in src_images.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    for img in images:
        link = out_images / img.name
        if not link.exists():
            link.symlink_to(img.resolve())

    counts = {"images": len(images), "bbox": 0, "polygon": 0, "skipped": 0}
    for label in src_labels.glob("*.txt"):
        lines = []
        for raw in label.read_text().splitlines():
            if not raw.strip():
                continue
            converted, kind = convert_line(raw)
            counts[kind] += 1
            if converted:
                lines.append(converted)
        (out_labels / label.name).write_text("\n".join(lines) + ("\n" if lines else ""))
    return counts


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="ROD-Dataset/dataset directory")
    p.add_argument("--split", default="test", choices=["train", "valid", "test"])
    p.add_argument("--model", default=None)
    p.add_argument("--work", default=str(REPO / "outputs" / "eval_data"))
    p.add_argument("--out", default=str(REPO / "outputs" / "eval"))
    p.add_argument("--device", default=None)
    p.add_argument("--batch", type=int, default=16)
    args = p.parse_args()

    from ultralytics import YOLO

    source, work, out = Path(args.dataset), Path(args.work), Path(args.out)
    model_path = resolve_model_path(args.model)
    counts = prepare_split(source, args.split, work)
    print(f"{args.split}: {counts}")

    split_key = "val" if args.split == "valid" else args.split
    data_yaml = work / "rod_eval.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(work.resolve()), "train": f"{args.split}/images", "val": f"{args.split}/images",
        "test": f"{args.split}/images", "nc": len(CLASS_NAMES), "names": CLASS_NAMES,
    }, sort_keys=False))

    model = YOLO(str(model_path))
    if list(model.names.values()) != CLASS_NAMES:
        raise RuntimeError("Checkpoint class names do not match configs/rod_detection.yaml")
    start = time.time()
    metrics = model.val(
        data=str(data_yaml), split=split_key, imgsz=640, batch=args.batch, device=args.device,
        project=str(out), name=f"rod_yolov8n_50ep_{args.split}", exist_ok=True, plots=True, verbose=True,
    )
    per_class = {
        CLASS_NAMES[c]: {"mAP50": round(float(metrics.box.ap50[i]), 4), "mAP50_95": round(float(metrics.box.ap[i]), 4)}
        for i, c in enumerate(metrics.box.ap_class_index)
    }
    result = {
        "model": model_path.name,
        "split": args.split,
        "label_preparation": counts,
        "images": counts["images"],
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "per_class": per_class,
        "eval_seconds": round(time.time() - start, 1),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / f"metrics_{args.split}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "per_class"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
