"""Build the fine-tuning dataset from the committed split lists.

Step 1, prepare-rod: materialise ROD train/val/test from configs/finetune_v1/splits/
into <work>/rod_<split>/{images,labels}. Images are symlinks; labels are converted
exactly as in the original training run (polygons -> boxes). The source is never modified.

Step 2, assemble (after coco_subset and label_completion): write the image lists and
<work>/data_v1.yaml for Ultralytics. Training = ROD train + COCO (capped so COCO is at
most --coco-max-frac of training images) + phone "train" walks. Validation = ROD val +
phone "val" walks. Test sets (ROD test, phone "test" walks) are written separately and
are never trained on.

Usage (Kaggle):
    python -m training.build_dataset prepare-rod --rod /kaggle/input/... --work /kaggle/working/ft_v1
    python -m training.build_dataset assemble --work /kaggle/working/ft_v1 [--phone /kaggle/input/<phone-dataset>]
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import yaml

from training.common import CLASS_NAMES, CONFIG_DIR, IMAGE_EXTENSIONS, SEED, SPLIT_DIR, convert_line, read_list

PHONE_WALKS = CONFIG_DIR / "phone_walks.yaml"


def prepare_rod(rod_root: Path, work: Path, split_dir: Path = SPLIT_DIR) -> dict:
    report = {}
    for split in ("train", "val", "test"):
        out = work / f"rod_{split}"
        (out / "images").mkdir(parents=True, exist_ok=True)
        (out / "labels").mkdir(parents=True, exist_ok=True)
        counts = {"images": 0, "bbox": 0, "polygon": 0, "skipped": 0, "missing_label_file": 0}
        for rel in read_list(split_dir / f"rod_{split}.txt"):
            src = rod_root / rel
            if not src.is_file():
                raise FileNotFoundError(f"Listed image missing: {src}")
            link = out / "images" / src.name
            if not link.exists():
                link.symlink_to(src.resolve())
            counts["images"] += 1
            label = src.parent.parent / "labels" / (src.stem + ".txt")
            lines = []
            if label.exists():
                for raw in label.read_text().splitlines():
                    if raw.strip():
                        converted, kind = convert_line(raw)
                        counts[kind] += 1
                        if converted:
                            lines.append(converted)
            else:
                counts["missing_label_file"] += 1
            (out / "labels" / (src.stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
        report[split] = counts
    return report


def images_in(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS) if folder.is_dir() else []


def phone_images(phone_root: Path | None, walks_file: Path = PHONE_WALKS) -> dict[str, list[Path]]:
    """Phone frames per split, from <phone_root>/<walk>/images/, using the walk assignment."""
    out = {"train": [], "val": [], "test": []}
    if phone_root is None:
        return out
    walks = yaml.safe_load(walks_file.read_text()) or {}
    seen = {}
    for split in out:
        for walk in walks.get(split) or []:
            if walk in seen:
                raise ValueError(f"Walk {walk!r} is assigned to both {seen[walk]} and {split}")
            seen[walk] = split
    for walk, split in seen.items():
        imgs = images_in(phone_root / walk / "images")
        if not imgs:
            raise FileNotFoundError(f"No images for walk {walk!r} in {phone_root / walk / 'images'}")
        unlabelled = [p.name for p in imgs if not (p.parent.parent / "labels" / (p.stem + ".txt")).exists()]
        if unlabelled:
            raise FileNotFoundError(
                f"Walk {walk!r}: {len(unlabelled)} images have no label file, e.g. {unlabelled[:3]}")
        out[split] += imgs
    return out


def cap_coco(rod_train: int, phone_train: int, coco: list[Path], max_frac: float, seed: int = SEED) -> list[Path]:
    """Largest fixed-seed sample of COCO images with coco / (all training images) <= max_frac."""
    if not 0 <= max_frac < 1:
        raise ValueError("coco-max-frac must be in [0, 1)")
    allowed = math.floor(max_frac * (rod_train + phone_train) / (1 - max_frac))
    if len(coco) <= allowed:
        return sorted(coco)
    return sorted(random.Random(seed).sample(sorted(coco), allowed))


def assemble(work: Path, phone_root: Path | None, coco_max_frac: float = 0.35, walks_file: Path = PHONE_WALKS) -> dict:
    rod = {s: images_in(work / f"rod_{s}" / "images") for s in ("train", "val", "test")}
    if not rod["train"]:
        raise FileNotFoundError(f"Run prepare-rod first: no images in {work / 'rod_train' / 'images'}")
    phone = phone_images(phone_root, walks_file)
    coco_train = cap_coco(len(rod["train"]), len(phone["train"]), images_in(work / "coco_train" / "images"),
                          coco_max_frac)
    lists = {
        "train": rod["train"] + coco_train + phone["train"],
        "val": rod["val"] + phone["val"],
        "test_rod": rod["test"],
        "test_phone": phone["test"],
        "test_coco": images_in(work / "coco_val" / "images"),
    }
    for name, imgs in lists.items():
        (work / f"{name}.txt").write_text("".join(f"{p}\n" for p in imgs))
    data = {"path": str(work.resolve()), "train": "train.txt", "val": "val.txt", "test": "test_rod.txt",
            "nc": len(CLASS_NAMES), "names": CLASS_NAMES}
    (work / "data_v1.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    # Run B control (configs/finetune_v1/train.yaml): ROD only, same validation set as run A.
    (work / "train_rod_only.txt").write_text("".join(f"{p}\n" for p in rod["train"]))
    (work / "data_v1_rod_only.yaml").write_text(
        yaml.safe_dump({**data, "train": "train_rod_only.txt"}, sort_keys=False))
    for name in ("test_rod", "test_phone", "test_coco"):
        if lists[name]:
            variant = {**data, "val": f"{name}.txt", "test": f"{name}.txt"}
            (work / f"data_{name}.yaml").write_text(yaml.safe_dump(variant, sort_keys=False))
    report = {name: len(imgs) for name, imgs in lists.items()}
    report["coco_share_of_train"] = round(len(coco_train) / max(1, len(lists["train"])), 3)
    report["phone_walks_used"] = bool(phone_root)
    (work / "dataset_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare-rod")
    a.add_argument("--rod", required=True, help="ROD-Dataset/dataset folder")
    a.add_argument("--work", required=True)
    b = sub.add_parser("assemble")
    b.add_argument("--work", required=True)
    b.add_argument("--phone", default=None, help="Folder with <walk>/images and <walk>/labels (optional)")
    b.add_argument("--coco-max-frac", type=float, default=0.35)
    args = p.parse_args(argv)
    if args.cmd == "prepare-rod":
        report = prepare_rod(Path(args.rod), Path(args.work))
    else:
        report = assemble(Path(args.work), Path(args.phone) if args.phone else None, args.coco_max_frac)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
