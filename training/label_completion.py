"""Fill in labels that each dataset leaves out, before fine-tuning.

ROD merges 26 sources, each labelling only its own classes, so many people and cars
are unlabelled; COCO never labels stairs, manholes, poles and other ROD classes.
Unlabelled objects teach a detector to ignore them. This adds high-confidence
model predictions as labels:

  - COCO images: the shipped ROD model adds ROD-only classes (conf >= 0.5).
  - ROD train/val images: the COCO model adds people and vehicles (conf >= 0.6) that
    do not overlap an existing person/vehicle label (IoU >= 0.5 or 80% containment,
    class-agnostic, so a ROD "Truck" or motorbike "Bike" is not labelled twice).

Test images are never touched: they stay human-labelled. Only the working copies
of label files are changed; every added box is logged, and a review sheet of random
added boxes is written so a person can check them before training (re-run with a
higher --rod-conf/--coco-conf if more than about 10% are wrong).

Usage (Kaggle, after build_dataset prepare-rod and coco_subset):
    python -m training.label_completion --work /kaggle/working/ft_v1 --rod-model models/rod_yolov8n_50ep_best.pt \
        --coco-model yolov8n.pt --device 0
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from training.common import (
    CLASS_ID, CLASS_NAMES, COMPLETE_IN_ROD, IMAGE_EXTENSIONS, ROD_ONLY_CLASSES, SEED, overlap, read_yolo_labels,
    to_xyxy, yolo_line,
)

COCO_NAME_FOR = {"Person": "person", "Car": "car", "Bus": "bus", "Truck": "truck", "Motorcycle": "motorcycle"}
# Existing ROD labels that already cover a person or vehicle (Bike includes motorbikes in ROD).
COVERING = {CLASS_ID[c] for c in ("Person", "Car", "Bus", "Truck", "Motorcycle", "Bike")}


def new_boxes(existing: list[tuple], predictions: list[tuple[int, float, tuple]], w: int, h: int,
              covering: set[int] | None, iou: float = 0.5, ios: float = 0.8) -> list[tuple[int, float, tuple]]:
    """Predictions (class_id, conf, xyxy) to add.

    With `covering=None`, every prediction is added: used for classes the dataset never labels,
    where overlapping other objects is normal (a person standing on stairs). Otherwise a prediction
    is skipped if it overlaps an existing label whose class is in `covering`, or one already added.
    """
    if covering is None:
        return sorted(predictions, key=lambda p: -p[1])
    taken = [to_xyxy(r, w, h) for r in existing if r[0] in covering]
    added = []
    for cls, conf, box in sorted(predictions, key=lambda p: -p[1]):
        if any(o[0] >= iou or o[1] >= ios for o in (overlap(box, t) for t in taken)):
            continue
        added.append((cls, conf, box))
        taken.append(box)
    return added


def complete(model, image_dir: Path, label_dir: Path, keep: dict[int, int], conf: float, covering: set[int] | None,
             device, log: list, source: str, batch: int = 32) -> Counter:
    """Run `model` on every image; add predictions of classes in `keep` (model id -> ROD id)."""
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    added = Counter()
    for result in model.predict(source=[str(p) for p in images], conf=conf, device=device, stream=True,
                                verbose=False, batch=batch):
        path = Path(result.path)
        h, w = result.orig_shape
        preds = [(keep[int(c)], float(s), tuple(map(float, b)))
                 for c, s, b in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxy.tolist())
                 if int(c) in keep]
        label_path = label_dir / (path.stem + ".txt")
        for cls, score, box in new_boxes(read_yolo_labels(label_path), preds, w, h, covering):
            line = yolo_line(cls, box, w, h)
            if line:
                with label_path.open("a") as f:
                    f.write(line + "\n")
                added[CLASS_NAMES[cls]] += 1
                log.append({"source": source, "image": str(path), "class": CLASS_NAMES[cls], "conf": round(score, 3),
                            "box": [round(v, 1) for v in box]})
    return added


def review_sheet(log: list[dict], out: Path, n: int = 200, seed: int = SEED) -> None:
    """Contact sheets of n random added boxes (50 per page) for a human check."""
    import cv2
    import numpy as np

    sample = random.Random(seed).sample(log, min(n, len(log)))
    (out / "review_sample.json").write_text(json.dumps(sample, indent=1))
    for page in range(0, len(sample), 50):
        tiles = []
        for i, item in enumerate(sample[page:page + 50], start=page + 1):
            img = cv2.imread(item["image"])
            if img is None:
                continue
            x1, y1, x2, y2 = map(int, item["box"])
            pad = int(0.2 * max(x2 - x1, y2 - y1, 10))
            ys, xs = max(0, y1 - pad), max(0, x1 - pad)
            crop = img[ys:y2 + pad, xs:x2 + pad].copy()
            cv2.rectangle(crop, (x1 - xs, y1 - ys), (x2 - xs, y2 - ys), (0, 128, 255), 2)
            crop = cv2.resize(crop, (200, 200))
            cv2.rectangle(crop, (0, 0), (200, 20), (0, 0, 0), -1)
            cv2.putText(crop, f"{i} {item['class']} {item['conf']:.2f}", (3, 14), 0, 0.45, (255, 255, 255), 1)
            tiles.append(crop)
        while len(tiles) % 10:
            tiles.append(np.zeros((200, 200, 3), np.uint8))
        rows = [np.hstack(tiles[i:i + 10]) for i in range(0, len(tiles), 10)]
        cv2.imwrite(str(out / f"review_sheet_{page // 50 + 1}.jpg"), np.vstack(rows))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work", required=True)
    p.add_argument("--rod-model", required=True)
    p.add_argument("--coco-model", required=True)
    p.add_argument("--rod-conf", type=float, default=0.5)
    p.add_argument("--coco-conf", type=float, default=0.6)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)
    from ultralytics import YOLO

    work = Path(args.work)
    rod, coco = YOLO(args.rod_model), YOLO(args.coco_model)
    if list(rod.names.values()) != CLASS_NAMES:
        raise RuntimeError("ROD model classes do not match configs/rod_detection.yaml")
    rod_keep = {CLASS_ID[c]: CLASS_ID[c] for c in ROD_ONLY_CLASSES}
    coco_ids = {n: i for i, n in coco.names.items()}
    coco_keep = {coco_ids[COCO_NAME_FOR[c]]: CLASS_ID[c] for c in COMPLETE_IN_ROD}

    log, report = [], {"settings": {"rod_conf": args.rod_conf, "coco_conf": args.coco_conf}}
    for split in ("train", "val"):
        d = work / f"coco_{split}"
        if d.is_dir():
            report[f"coco_{split}: added by ROD model"] = complete(
                rod, d / "images", d / "labels", rod_keep, args.rod_conf, None, args.device, log, f"coco_{split}")
        d = work / f"rod_{split}"
        if d.is_dir():
            report[f"rod_{split}: added by COCO model"] = complete(
                coco, d / "images", d / "labels", coco_keep, args.coco_conf, COVERING, args.device, log, f"rod_{split}")
    with (work / "label_completion_added.jsonl").open("w") as f:
        f.writelines(json.dumps(item) + "\n" for item in log)
    report = {k: dict(v) if isinstance(v, Counter) else v for k, v in report.items()}
    report["total_added"] = len(log)
    (work / "label_completion_report.json").write_text(json.dumps(report, indent=2) + "\n")
    review_sheet(log, work)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
