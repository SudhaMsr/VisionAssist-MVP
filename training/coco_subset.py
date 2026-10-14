"""Select a street-like COCO 2017 subset and write ROD-format YOLO labels.

Kept: images with a car, bus, truck or motorcycle, or with a person plus an outdoor
cue (traffic light, stop sign, parking meter, fire hydrant or bench).
Skipped: images with a bicycle (ROD "Bike" is left unchanged in v1, so a bicycle could
not be labelled consistently) and images with "crowd" regions (unlabelled people).
Every COCO category with a ROD equivalent is mapped (training/common.py COCO_TO_ROD),
so no mapped object is left unlabelled. A fixed-seed sample keeps it reproducible.

Usage (Kaggle):
    python -m training.coco_subset --coco /kaggle/input --split train --limit 10000 --work /kaggle/working/ft_v1
Writes <work>/coco_<split>/{images,labels}/ (images are symlinks) and coco_<split>.txt.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from training.common import CLASS_ID, COCO_EXCLUDE_IMAGES_WITH, COCO_TO_ROD, SEED, find_one, yolo_line

VEHICLES = {"car", "bus", "truck", "motorcycle"}
OUTDOOR_CUES = {"traffic light", "stop sign", "parking meter", "fire hydrant", "bench"}


def select(coco: dict, limit: int | None, seed: int = SEED) -> tuple[list[dict], dict[int, list[dict]], dict]:
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    anns: dict[int, list[dict]] = {}
    for a in coco["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    chosen, why = [], Counter()
    for img in coco["images"]:
        names = {cat_name[a["category_id"]] for a in anns.get(img["id"], [])}
        if any(a.get("iscrowd") for a in anns.get(img["id"], [])):
            why["skipped: crowd region"] += 1
        elif names & COCO_EXCLUDE_IMAGES_WITH:
            why["skipped: contains bicycle"] += 1
        elif names & VEHICLES or ("person" in names and names & OUTDOOR_CUES):
            chosen.append(img)
        else:
            why["skipped: not street-like"] += 1
    chosen.sort(key=lambda i: i["id"])
    random.Random(seed).shuffle(chosen)
    eligible = len(chosen)
    chosen = sorted(chosen[:limit] if limit else chosen, key=lambda i: i["file_name"])
    stats = {"images_in_annotations": len(coco["images"]), "eligible": eligible, "selected": len(chosen), **why}
    return chosen, anns, stats


def labels_for(img: dict, anns: list[dict], cat_name: dict[int, str]) -> list[str]:
    lines = []
    for a in anns:
        rod = COCO_TO_ROD.get(cat_name[a["category_id"]])
        if rod:
            x, y, w, h = a["bbox"]
            line = yolo_line(CLASS_ID[rod], (x, y, x + w, y + h), img["width"], img["height"])
            if line:
                lines.append(line)
    return lines


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coco", required=True, help="Folder containing coco2017/ (searched recursively)")
    p.add_argument("--split", choices=["train", "val"], required=True)
    p.add_argument("--limit", type=int, default=None, help="Images to keep (fixed-seed sample)")
    p.add_argument("--work", required=True, help="Output folder")
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args(argv)

    ann_path = find_one(Path(args.coco), f"coco2017/annotations/instances_{args.split}2017.json")
    image_dir = ann_path.parent.parent / f"{args.split}2017"
    coco = json.loads(ann_path.read_text())
    chosen, anns, stats = select(coco, args.limit, args.seed)
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}

    out = Path(args.work) / f"coco_{args.split}"
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    boxes = Counter()
    for img in chosen:
        src = image_dir / img["file_name"]
        link = out / "images" / img["file_name"]
        if not link.exists():
            link.symlink_to(src)
        lines = labels_for(img, anns.get(img["id"], []), cat_name)
        boxes.update(int(line.split()[0]) for line in lines)
        (out / "labels" / (Path(img["file_name"]).stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
    (Path(args.work) / f"coco_{args.split}.txt").write_text("\n".join(i["file_name"] for i in chosen) + "\n")
    from training.common import CLASS_NAMES
    stats["boxes_by_rod_class"] = {CLASS_NAMES[c]: n for c, n in sorted(boxes.items())}
    (Path(args.work) / f"coco_{args.split}_report.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
